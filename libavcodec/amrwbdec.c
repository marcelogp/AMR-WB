/*
 * AMR wideband decoder
 * Copyright (c) 2010 Marcelo Galvao Povoa
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A particular PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "avcodec.h"
#include "get_bits.h"
#include "lsp.h"
#include "acelp_filters.h"
#include "acelp_vectors.h"

#include "amrwbdata.h"

/** Get x bits in the index interval [lsb,lsb+len-1] inclusive */
#define BIT_STR(x,lsb,len) (((x) >> (lsb)) & ((1 << (len)) - 1))

/** Get the bit at specified position */
#define BIT_POS(x, p) (((x) >> (p)) & 1)

typedef struct {
    AMRWBFrame                             frame; ///< AMRWB parameters decoded from bitstream
    enum Mode                        fr_cur_mode; ///< mode index of current frame
    uint8_t                           fr_quality; ///< frame quality index (FQI)
    uint8_t                          fr_mode_ind; ///< mode indication field
    uint8_t                          fr_mode_req; ///< mode request field
    uint8_t                               fr_crc; ///< crc for class A bits
    float                    isf_quant[LP_ORDER]; ///< quantized ISF vector from current frame
    float                   isf_q_past[LP_ORDER]; ///< quantized ISF vector of the previous frame
    double                      isp[4][LP_ORDER]; ///< ISP vectors from current frame
    double               isp_sub4_past[LP_ORDER]; ///< ISP vector for the 4th subframe of the previous frame
    
    float                   lp_coef[4][LP_ORDER]; ///< Linear Prediction Coefficients from ISP vector

    uint8_t                       base_pitch_lag; ///< integer part of pitch lag for next relative subframe
    
    float excitation_buf[PITCH_MAX + LP_ORDER + 1 + AMRWB_SUBFRAME_SIZE]; ///< current excitation and all necessary excitation history
    float                            *excitation; ///< points to current excitation in excitation_buf[]
    
    float      pitch_vector[AMRWB_SUBFRAME_SIZE]; ///< adaptive codebook (pitch) vector for current subframe
    
    float                          pitch_gain[5]; ///< quantified pitch gains for the current and previous four subframes 
} AMRWBContext;

static int amrwb_decode_init(AVCodecContext *avctx) 
{
    AMRWBContext *ctx = avctx->priv_data;
    int i;
    
    ctx->excitation = &ctx->excitation_buf[PITCH_MAX + LP_ORDER + 1];

    for (i = 0; i < LP_ORDER; i++) {
        ctx->isf_q_past[i]    = isf_init[i] / (float) (1 << 15);
        ctx->isp_sub4_past[i] = isp_init[i] / (float) (1 << 15);
    }
    
    return 0;
}

/**
 * Parses a speech frame, storing data in the Context
 * 
 * @param c                 [in/out] the context
 * @param buf               [in] pointer to the input buffer
 * @param buf_size          [in] size of the input buffer
 *
 * @return the frame mode
 */
static enum Mode unpack_bitstream(AMRWBContext *ctx, const uint8_t *buf,
                                  int buf_size)
{
    GetBitContext gb;
    enum Mode mode;
    uint16_t *data;
    
    init_get_bits(&gb, buf, buf_size * 8);
    
    /* AMR-WB header */
    ctx->fr_cur_mode  = get_bits(&gb, 4);
    mode              = ctx->fr_cur_mode;
    ctx->fr_quality   = get_bits1(&gb);
    
    skip_bits(&gb, 3);
    
    /* AMR-WB Auxiliary Information */
    ctx->fr_mode_ind = get_bits(&gb, 4);
    ctx->fr_mode_req = get_bits(&gb, 4);
    //XXX: Need to check conformity in mode_ind/mode_req and crc?
    ctx->fr_crc = get_bits(&gb, 8);
    
    data = (uint16_t *) &ctx->frame;
    memset(data, 0, sizeof(AMRWBFrame));
    buf++;
    
    if (mode < MODE_SID) { /* Normal speech frame */
        const uint16_t *perm = amr_bit_orderings_by_mode[mode];
        int field_size;
        
        while ((field_size = *perm++)) {
            int field = 0;
            int field_offset = *perm++;
            while (field_size--) {
               uint16_t bit = *perm++;
               field <<= 1;
               field |= buf[bit >> 3] >> (bit & 7) & 1;
            }
            data[field_offset] = field;
        }
    }
    else if (mode == MODE_SID) { /* Comfort noise frame */
        /* not implemented */
    }
    
    return mode;
}

/**
 * Convert an ISF vector into an ISP vector.
 *
 * @param isf               input isf vector
 * @param isp               output isp vector
 */
static void isf2isp(const float *isf, double *isp)
{
    int i;

    for (i = 0; i < LP_ORDER; i++)
        isp[i] = cos(2.0 * M_PI * isf[i]);
}

/**
 * Decodes quantized ISF vectors using 36-bit indices (6K60 mode only) 
 * 
 * @param ind               [in] array of 5 indices
 * @param isf_q             [out] isf_q[LP_ORDER]
 * @param fr_q              [in] frame quality (good frame == 1)
 *
 */
static void decode_isf_indices_36b(uint16_t *ind, float *isf_q, uint8_t fr_q) {
    int i;
    
    if (fr_q == 1) {
        for (i = 0; i < 9; i++) {
            isf_q[i] = dico1_isf[ind[0]][i] / (float) (1<<15);
        }
        for (i = 0; i < 7; i++) {
            isf_q[i + 9] = dico2_isf[ind[1]][i] / (float) (1<<15);
        }
        for (i = 0; i < 5; i++) {
            isf_q[i] = isf_q[i] + dico21_isf_36b[ind[2]][i] / (float) (1<<15);
        }
        for (i = 0; i < 4; i++) {
            isf_q[i + 5] = isf_q[i + 5] + dico22_isf_36b[ind[3]][i] / (float) (1<<15);
        }
        for (i = 0; i < 7; i++) {
            isf_q[i + 9] = isf_q[i + 9] + dico23_isf_36b[ind[4]][i] / (float) (1<<15);
        }
    }
    /* not implemented for bad frame */
}

/**
 * Decodes quantized ISF vectors using 46-bit indices (except 6K60 mode) 
 * 
 * @param ind                 [in] array of 7 indices
 * @param isf_q               [out] isf_q[LP_ORDER]
 * @param fr_q                [in] frame quality (good frame == 1)
 *
 */
static void decode_isf_indices_46b(uint16_t *ind, float *isf_q, uint8_t fr_q) {
    int i;

    if (fr_q == 1) {
        for (i = 0; i < 9; i++) {
            isf_q[i] = dico1_isf[ind[0]][i] / (float) (1<<15);
        }
        for (i = 0; i < 7; i++) {
            isf_q[i + 9] = dico2_isf[ind[1]][i] / (float) (1<<15);
        }
        for (i = 0; i < 3; i++) {
            isf_q[i] = isf_q[i] + dico21_isf[ind[2]][i] / (float) (1<<15);
        }
        for (i = 0; i < 3; i++) {
            isf_q[i + 3] = isf_q[i + 3] + dico22_isf[ind[3]][i] / (float) (1<<15);
        }
        for (i = 0; i < 3; i++) {
            isf_q[i + 6] = isf_q[i + 6] + dico23_isf[ind[4]][i] / (float) (1<<15);    
        }
        for (i = 0; i < 3; i++) {
            isf_q[i + 9] = isf_q[i + 9] + dico24_isf[ind[5]][i] / (float) (1<<15);
        }
        for (i = 0; i < 4; i++) {
            isf_q[i + 12] = isf_q[i + 12] + dico25_isf[ind[6]][i] / (float) (1<<15);
        }
    }
    /* not implemented for bad frame */
}

/**
 * Apply mean and past ISF values using the predicion factor 
 * Updates past ISF vector
 * 
 * @param isf_q               [in] current quantized ISF
 * @param isf_past            [in/out] past quantized ISF
 *
 */
static void isf_add_mean_and_past(float *isf_q, float *isf_past) {
    int i;
    float tmp;
    
    for (i = 0; i < LP_ORDER; i++) {
        tmp = isf_q[i];
        isf_q[i] = tmp + isf_mean[i] / (float) (1<<15);
        isf_q[i] = isf_q[i] + PRED_FACTOR * isf_past[i];
        isf_past[i] = tmp;
    }
}

/**
 * Ensures a minimum distance between adjacent ISFs
 * 
 * @param isf                 [in/out] ISF vector
 * @param min_spacing         [in] minimum gap to keep
 * @param size                [in] ISF vector size
 *
 */
static void isf_set_min_dist(float *isf, float min_spacing, int size) {
    int i;
    float prev = 0.0;
    
    for (i = 0; i < size; i++) {
        isf[i] = FFMAX(isf[i], prev + min_spacing);
        prev = isf[i];
    }
}

/**
 * Interpolate the fourth ISP vector from current and past frame
 * to obtain a ISP vector for each subframe
 *
 * @param isp_q               [in/out] ISPs for each subframe
 * @param isp4_past           [in] Past ISP for subframe 4
 */
static void interpolate_isp(double isp_q[4][LP_ORDER], double *isp4_past)
{
    int i;
    /* XXX: Did not used ff_weighted_vector_sumf because using double */
    
    for (i = 0; i < LP_ORDER; i++)
        isp_q[0][i] = 0.55 * isp4_past[i] + 0.45 * isp_q[3][i];
        
    for (i = 0; i < LP_ORDER; i++)
        isp_q[1][i] = 0.20 * isp4_past[i] + 0.80 * isp_q[3][i];
        
    for (i = 0; i < LP_ORDER; i++)
        isp_q[2][i] = 0.04 * isp4_past[i] + 0.96 * isp_q[3][i];
}

/**
 * Convert a ISP vector to LP coefficient domain {a_k}
 * Equations from TS 26.190 section 5.2.4
 *
 * @param isp                 [in] ISP vector for a subframe
 * @param lp                  [out] LP coefficients
 * @param lp_half_order       [in] Half the number of LPs to construct
 */
static void isp2lp(double isp[LP_ORDER], float *lp, int lp_half_order) {
    double pa[MAX_LP_HALF_ORDER+1], qa[MAX_LP_HALF_ORDER+1];
    float *lp2 = lp + (lp_half_order << 1);
    double last_isp = isp[2 * lp_half_order - 1];
    double qa_old = 0; /* XXX: qa[i-2] assuming qa[-1] = 0, not mentioned in document */ 
    int i;
    
    ff_lsp2polyf(isp,     pa, lp_half_order);
    ff_lsp2polyf(isp + 1, qa, lp_half_order);
    
    for (i=1; i<lp_half_order; i++) {
        double paf = (1 + last_isp) * pa[i];
        double qaf = (1 - last_isp) * (qa[i] - qa_old);
        
        qa_old = qa[i-1];
        
        lp[i]  = 0.5 * (paf + qaf);
        lp2[i] = 0.5 * (paf - qaf);
    }
    
    lp2[0] = 0.5 * (1 + last_isp) * pa[lp_half_order] * lp_half_order;
    lp2[lp_half_order] = last_isp;
}

/**
 * Decode a adaptive codebook index into pitch lag (except 6k60, 8k85 modes)
 * Calculate (nearest) integer lag and fractional lag always using 1/4 resolution
 * In 1st and 3rd subframes index is relative to last subframe integer lag
 *
 * @param lag_int             [out] Decoded integer pitch lag
 * @param lag_frac            [out] Decoded fractional pitch lag
 * @param pitch_index         [in] Adaptive codebook pitch index
 * @param base_lag_int        [in/out] Base integer lag used in relative subframes
 * @param subframe            [in] Current subframe index (0 to 3)
 */
static void decode_pitch_lag_high(int *lag_int, int *lag_frac, int pitch_index,
                                  uint8_t *base_lag_int, const int subframe)
{
    if (subframe == 0 || subframe == 2) {
        if (pitch_index < 376) {
            *lag_int  = (pitch_index + 137) >> 2;
            *lag_frac = pitch_index - (*lag_int << 2) + 136;
        } else if (pitch_index < 440) {
            *lag_int  = (pitch_index + 257 - 376) >> 1;
            *lag_frac = (pitch_index - (*lag_int << 1) + 256 - 376) << 1;
            /* the actual resolution is 1/2 but expressed as 1/4 */
        } else {
            *lag_int  = pitch_index - 280;
            *lag_frac = 0;
        }
        *base_lag_int = *lag_int; // store previous lag
    } else {
        *lag_int  = (pitch_index + 1) >> 2;
        *lag_frac = pitch_index - (*lag_int << 2);
        *lag_int += *base_lag_int - 8;
        /* XXX: Doesn't seem to need bounding according to TS 26.190 */ 
    }
}

/**
 * Decode a adaptive codebook index into pitch lag for 8k85 mode
 * Description is analogous to decode_pitch_lag_high
 */
static void decode_pitch_lag_8K85(int *lag_int, int *lag_frac, int pitch_index,
                                  uint8_t *base_lag_int, const int subframe)
{
    if (subframe == 0 || subframe == 2) {
        if (pitch_index < 116) {
            *lag_int  = (pitch_index + 69) >> 1;
            *lag_frac = (pitch_index - (*lag_int << 1) + 68) << 1;
        } else {
            *lag_int  = pitch_index - 24;
            *lag_frac = 0;
        }
        *base_lag_int = *lag_int;
    } else {
        *lag_int  = (pitch_index + 1) >> 1;
        *lag_frac = pitch_index - (*lag_int << 1);
        *lag_int += *base_lag_int - 8;
    }
}

/**
 * Decode a adaptive codebook index into pitch lag for 6k60 mode
 * Description is analogous to decode_pitch_lag_high, but relative
 * index is used for all subframes except the first
 */
static void decode_pitch_lag_6K60(int *lag_int, int *lag_frac, int pitch_index,
                                  uint8_t *base_lag_int, const int subframe)
{
    if (subframe == 0) {
        if (pitch_index < 116) {
            *lag_int  = (pitch_index + 69) >> 1;
            *lag_frac = (pitch_index - (*lag_int << 1) + 68) << 1;
        } else {
            *lag_int  = pitch_index - 24;
            *lag_frac = 0;
        }
        *base_lag_int = *lag_int;
    } else {
        *lag_int  = (pitch_index + 1) >> 1;
        *lag_frac = pitch_index - (*lag_int << 1);
        *lag_int += *base_lag_int - 8;
    }
}

static void decode_pitch_vector(AMRWBContext *ctx,
                                const AMRWBSubFrame *amr_subframe,
                                const int subframe)
{
    int pitch_lag_int, pitch_lag_frac;
    int i;
    float *exc = ctx->excitation;
    enum Mode mode = ctx->fr_cur_mode;
    
    if (mode == MODE_6k60) {
        decode_pitch_lag_6K60(&pitch_lag_int, &pitch_lag_frac, amr_subframe->adap,
                              &ctx->base_pitch_lag, subframe);
    } else if (mode == MODE_8k85) {
        decode_pitch_lag_8K85(&pitch_lag_int, &pitch_lag_frac, amr_subframe->adap,
                              &ctx->base_pitch_lag, subframe);
    } else
        decode_pitch_lag_high(&pitch_lag_int, &pitch_lag_frac, amr_subframe->adap,
                              &ctx->base_pitch_lag, subframe);
                              
     pitch_lag_int += pitch_lag_frac > 0;
     
    /* Calculate the pitch vector by interpolating the past excitation at the
       pitch lag using a hamming windowed sinc function. */
    ff_acelp_interpolatef(exc, exc + 1 - pitch_lag_int,
                          ac_inter, 4,
                          pitch_lag_frac + 4 - 4*(pitch_lag_frac > 0),
                          LP_ORDER, AMRWB_SUBFRAME_SIZE);

    /* Check which pitch signal path should be used */
    if (amr_subframe->ltp) {
        memcpy(ctx->pitch_vector, exc, AMRWB_SUBFRAME_SIZE * sizeof(float));
    } else {
        for (i = 0; i < AMRWB_SUBFRAME_SIZE; i++)
            ctx->pitch_vector[i] = 0.18 * exc[i - 1] + 0.64 * exc[i] + 0.18 * exc[i + 1];
    }
}

/**
 * The next six functions decode_[i]p_track decode exactly i pulses
 * positions and amplitudes (-1 or 1) in a subframe track using
 * an encoded pulse indexing (TS 26.190 section 5.8.2).
 *
 * The results are given in out[], in which a negative number means
 * amplitude -1 and vice-versa. (i.e., ampl = x/abs(x) )
 *
 * @param out                 [out] Output buffer (writes i elements)
 * @param code                [in] Pulse index (no. of bits varies, see below)
 * @param m                   [in] (log2) Number of potential positions
 * @param off                 [in] Offset for decoded positions
 */
//XXX: Some of these functions are simple and recurrent (used inline)

static inline void decode_1p_track(int *out, int code, int m, int off)
{   ///code: m+1 bits
    int pos = BIT_STR(code, 0, m) + off;
    
    out[0] = BIT_POS(code, m) ? -pos : pos;
}

static inline void decode_2p_track(int *out, int code, int m, int off)
{   ///code: 2m+1 bits
    int pos0 = BIT_STR(code, m, m) + off;
    int pos1 = BIT_STR(code, 0, m) + off;
    
    out[0] = BIT_POS(code, 2*m) ? -pos0 : pos0;
    out[1] = BIT_POS(code, 2*m) ? -pos1 : pos1;
    out[1] = pos0 > pos1 ? -out[1] : out[1];
}

static void decode_3p_track(int *out, int code, int m, int off)
{   ///code: 3m+1 bits
    int half_2p = BIT_POS(code, 2*m - 1) << (m - 1);
    
    decode_2p_track(out, BIT_STR(code, 0, 2*m - 1),
                    m - 1, off + half_2p);
    decode_1p_track(out + 2, BIT_STR(code, 2*m, m + 1), m, off);
}

static void decode_4p_track(int *out, int code, int m, int off)
{   ///code: 4m bits
    int half_4p, subhalf_2p;
    int b_offset = 1 << (m - 1);
    
    switch (BIT_STR(code, 4*m - 2, 2)) /* case ID (2 bits) */
    {
        case 0: /* 0 pulses in A, 4 pulses in B or vice-versa */
            half_4p = BIT_POS(code, 4*m - 3) << (m - 1); /* which has 4 pulses */
            subhalf_2p = BIT_POS(code, 2*m - 3) << (m - 2);
            
            decode_2p_track(out, BIT_STR(code, 0, 2*m - 3),
                            m - 2, off + half_4p + subhalf_2p);
            decode_2p_track(out + 2, BIT_STR(code, 2*m - 2, 2*m - 1),
                            m - 1, off + half_4p);
            break;
        case 1: /* 1 pulse in A, 3 pulses in B */
            decode_1p_track(out, BIT_STR(code,  3*m - 2, m),
                            m - 1, off);
            decode_3p_track(out + 1, BIT_STR(code, 0, 3*m - 2),
                            m - 1, off + b_offset);
            break;
        case 2: /* 2 pulses in each half */
            decode_2p_track(out, BIT_STR(code, 2*m - 1, 2*m - 1),
                            m - 1, off);
            decode_2p_track(out + 2, BIT_STR(code, 0, 2*m - 1),
                            m - 1, off + b_offset);
            break;
        case 3: /* 3 pulses in A, 1 pulse in B */
            decode_3p_track(out, BIT_STR(code, m, 3*m - 2),
                            m - 1, off);
            decode_1p_track(out + 3, BIT_STR(code, 0, m),
                            m - 1, off + b_offset);
            break;
    }
}

static void decode_5p_track(int *out, int code, int m, int off)
{   ///code: 5m bits
    int half_3p = BIT_POS(code, 5*m - 1) << (m - 1);
    
    decode_3p_track(out, BIT_STR(code, 2*m, 3*m - 2),
                    m - 1, off + half_3p);
    //XXX: there seems to be a typo in I3p expoent (from reference)
    decode_2p_track(out + 3, BIT_STR(code, 0, 2*m + 1), m, off);
}

static void decode_6p_track(int *out, int code, int m, int off)
{   ///code: 6m-2 bits
    int b_offset = 1 << (m - 1);
    /* which half has more pulses in cases 0 to 2 */
    int half_more  = BIT_POS(code, 6*m - 5) << (m - 1);
    int half_other = b_offset - half_more;
    
    switch (BIT_STR(code, 6*m - 4, 2)) /* case ID (2 bits) */
    {
        case 0: /* 0 pulses in A, 6 pulses in B or vice-versa */
            decode_1p_track(out, BIT_STR(code, 0, m),
                            m - 1, off + half_more);
            decode_5p_track(out + 1, BIT_STR(code, m, 5*m - 5),
                            m - 1, off + half_more);
            break;
        case 1: /* 1 pulse in A, 5 pulses in B or vice-versa */
            decode_1p_track(out, BIT_STR(code, 0, m),
                            m - 1, off + half_other);
            decode_5p_track(out + 1, BIT_STR(code, m, 5*m - 5),
                            m - 1, off + half_more);
            break;
        case 2: /* 2 pulses in A, 4 pulses in B or vice-versa */
            decode_2p_track(out, BIT_STR(code, 0, 2*m - 1),
                            m - 1, off + half_other);
            decode_4p_track(out + 2, BIT_STR(code, 2*m - 1, 4*m - 4),
                            m - 1, off + half_more);
            break;
        case 3: /* 3 pulses in A, 3 pulses in B */
            decode_3p_track(out, BIT_STR(code, 3*m - 2, 3*m - 2),
                            m - 1, off);
            decode_3p_track(out + 3, BIT_STR(code, 0, 3*m - 2),
                            m - 1, off + b_offset);
            break;
    }
}

/**
 * Decode the algebraic codebook index to pulse positions and signs,
 * then construct the algebraic codebook vector.
 *
 * @param fixed_sparse        [out] pointer to the algebraic (innovative) codebook
 * @param pulse_hi            [in] MSBs part of the pulse index array (higher modes only)
 * @param pulse_lo            [in] LSBs part of the pulse index array
 * @param mode                [in] mode of the current frame
 */
// XXX: For now, uses the same AMRFixed struct from AMR-NB but
// the maximum number of pulses in it was increased to 24
static void decode_fixed_sparse(AMRFixed *fixed_sparse, const uint16_t *pulse_hi,
                                const uint16_t *pulse_lo, const enum Mode mode)
{
    /* sig_pos stores for each track the decoded pulse position
     * indexes multiplied by its corresponding amplitude (+1 or -1) */
    int sig_pos[4][6];
    int pulses_nb = 0;
    int spacing = (mode == MODE_6k60) ? 2 : 4;
    int i, j;
    
    switch (mode) {
        case MODE_6k60:
            for (i = 0; i < 2; i++)
                decode_1p_track(sig_pos[i], pulse_lo[i], 5, 0);
            break;
        case MODE_8k85:
            for (i = 0; i < 4; i++)
                decode_1p_track(sig_pos[i], pulse_lo[i], 4, 0);
            break;
        case MODE_12k65:
            for (i = 0; i < 4; i++)
                decode_2p_track(sig_pos[i], pulse_lo[i], 4, 0);
            break;
        case MODE_14k25:
            for (i = 0; i < 2; i++)
                decode_3p_track(sig_pos[i], pulse_lo[i], 4, 0);
            for (i = 2; i < 4; i++)
                decode_2p_track(sig_pos[i], pulse_lo[i], 4, 0);
            break;
        case MODE_15k85:
            for (i = 0; i < 4; i++)
                decode_3p_track(sig_pos[i], pulse_lo[i], 4, 0);
            break;
        case MODE_18k25:
            for (i = 0; i < 4; i++)
                decode_4p_track(sig_pos[i], (int) pulse_lo[i] +
                               ((int) pulse_hi[i] << 14), 4, 0);
            break;
        case MODE_19k85:
            for (i = 0; i < 2; i++)
                decode_5p_track(sig_pos[i], (int) pulse_lo[i] +
                               ((int) pulse_hi[i] << 10), 4, 0);
            for (i = 2; i < 4; i++)
                decode_4p_track(sig_pos[i], (int) pulse_lo[i] +
                               ((int) pulse_hi[i] << 14), 4, 0);
            break;
        case MODE_23k05:
        case MODE_23k85:
            for (i = 0; i < 4; i++)
                decode_6p_track(sig_pos[i], (int) pulse_lo[i] +
                               ((int) pulse_hi[i] << 11), 4, 0);
            break;
    }

    for (i = 0; i < 4; i++)
        for (j = 0; j < pulses_nb_per_mode_tr[mode][i]; j++) {
            int pos = sig_pos[i][j];
            fixed_sparse->x[pulses_nb] = FFABS(pos) * spacing + i;
            fixed_sparse->y[pulses_nb] = pos < 0 ? -1.0 : 1.0;
            pulses_nb++;
        }
    
    fixed_sparse->n = pulses_nb;
}

/**
 * Decode pitch gain and fixed gain correction factor
 *
 * @param vq_gain             [in] vector-quantized index for gains
 * @param mode                [in] mode of the current frame
 * @param fixed_gain_factor   [out] decoded fixed gain correction factor
 * @param pitch_gain          [out] decoded pitch gain
 */
static void decode_gains(const uint8_t vq_gain, const enum Mode mode,
                         float *fixed_gain_factor, float *pitch_gain)
{
    const int16_t *gains;
    
    if (mode == MODE_6k60 || mode == MODE_8k85)
        gains = qua_gain_6b[vq_gain];
    else
        gains = qua_gain_7b[vq_gain];
    
    *pitch_gain        = gains[0] * (1.0 / 16384.0);
    *fixed_gain_factor = gains[1] * (1.0 / 2048.0);
}

static int amrwb_decode_frame(AVCodecContext *avctx, void *data, int *data_size,
                              AVPacket *avpkt)
{
    AMRWBContext *ctx  = avctx->priv_data;
    AMRWBFrame   *cf   = &ctx->frame;  
    const uint8_t *buf = avpkt->data;
    int buf_size       = avpkt->size;
    AMRFixed fixed_sparse = {0};             // fixed vector up to anti-sparseness processing
    float fixed_gain_factor;                 // fixed gain correction factor (gamma)
    int sub;
    
    ctx->fr_cur_mode = unpack_bitstream(ctx, buf, buf_size);
    
    if (ctx->fr_cur_mode == MODE_SID) {
        av_log_missing_feature(avctx, "SID mode", 1);
        return -1;
    }
    if (!ctx->fr_quality) {
        av_log(avctx, AV_LOG_ERROR, "Encountered a bad or corrupted frame\n");
    }
    
    /* Decode the quantized ISF vector */
    if (ctx->fr_cur_mode == MODE_6k60) {
        decode_isf_indices_36b(cf->isp_id, ctx->isf_quant, ctx->fr_quality);
    }
    else {
        decode_isf_indices_46b(cf->isp_id, ctx->isf_quant, ctx->fr_quality);
    }
    
    isf_add_mean_and_past(ctx->isf_quant, ctx->isf_q_past);
    isf_set_min_dist(ctx->isf_quant, MIN_ISF_SPACING, LP_ORDER);
    
    isf2isp(ctx->isf_quant, ctx->isp[3]);
    /* Generate a ISP vector for each subframe */
    interpolate_isp(ctx->isp, ctx->isp_sub4_past);
    
    for (sub = 0; sub < 4; sub++)
        isp2lp(ctx->isp[sub], ctx->lp_coef[sub], LP_ORDER/2);
        
    for (sub = 0; sub < 4; sub++) {
        const AMRWBSubFrame *cur_subframe = &cf->subframe[sub];

        /* Decode adaptive codebook */
        decode_pitch_vector(ctx, cur_subframe, sub);
        /* Decode innovative codebook (sparse representation) */
        decode_fixed_sparse(&fixed_sparse, cur_subframe->pul_ih,
                            cur_subframe->pul_il, ctx->fr_cur_mode);

        decode_gains(cur_subframe->vq_gain, ctx->fr_cur_mode,
                     &fixed_gain_factor, &ctx->pitch_gain[4]);
    }
    
    //update state for next frame
    memcpy(ctx->isp_sub4_past, ctx->isp[3], LP_ORDER * sizeof(ctx->isp[3][0])); 
    
    return 0;
}

AVCodec amrwb_decoder =
{
    .name           = "amrwb",
    .type           = CODEC_TYPE_AUDIO,
    .id             = CODEC_ID_AMR_WB,
    .priv_data_size = sizeof(AMRWBContext),
    .init           = amrwb_decode_init,
    .decode         = amrwb_decode_frame,
    .long_name      = NULL_IF_CONFIG_SMALL("Adaptive Multi-Rate WideBand"),
};
