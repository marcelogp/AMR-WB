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

#include "amrwbdata.h"

typedef struct {
    AMRWBFrame          frame;                      ///< AMRWB parameters decoded from bitstream
    enum Mode           fr_cur_mode;                ///< mode index of current frame
    uint8_t             fr_quality;                 ///< frame quality index (FQI)
    uint8_t             fr_mode_ind;                ///< mode indication field
    uint8_t             fr_mode_req;                ///< mode request field
    uint8_t             fr_crc;                     ///< crc for class A bits
    float               isf_quant[LP_ORDER];        ///< quantized ISF vector from current frame
    float               isf_q_past[LP_ORDER];       ///< quantized ISF vector of the previous frame
    double              isp[4][LP_ORDER];           ///< ISP vectors from current frame
    double              isp_sub4_past[LP_ORDER];    ///< ISP vector for the 4th subframe of the previous frame

} AMRWBContext;

static int amrwb_decode_init(AVCodecContext *avctx) 
{
    AMRWBContext *ctx = avctx->priv_data;
    int i;

    for (i = 0; i < LP_ORDER; i++)
        ctx->isf_q_past[i] = isf_init[i] / (float) (1 << 15);
    
    return 0;
}

/**
 * Parses a speech frame, storing data in the Context
 * 
 * @param c                 the context
 * @param buf               pointer to the input buffer
 * @param buf_size          size of the input buffer
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
    ///Need to check conformity in mode_ind/mode_req and crc?
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
 * @param ind               [in] array of 7 indices
 * @param isf_q             [out] isf_q[LP_ORDER]
 * @param fr_q              [in] frame quality (good frame == 1)
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
        isf_q[i] = tmp + isf_mean[i];
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

static int amrwb_decode_frame(AVCodecContext *avctx, void *data, int *data_size,
                              AVPacket *avpkt)
{
    AMRWBContext *ctx  = avctx->priv_data;
    AMRWBFrame   *cf   = &ctx->frame;  
    const uint8_t *buf = avpkt->data;
    int buf_size       = avpkt->size;
    
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
    
    //isf2isp(ctx->isf_quant, ctx->isp[3]);
    
    return 0;
}

static int amrwb_decode_close(AVCodecContext *avctx)
{
    return 0;
}

AVCodec amrwb_decoder =
{
    .name           = "amrwb",
    .type           = CODEC_TYPE_AUDIO,
    .id             = CODEC_ID_AMR_WB,
    .priv_data_size = sizeof(AMRWBContext),
    .init           = amrwb_decode_init,
    .close          = amrwb_decode_close,
    .decode         = amrwb_decode_frame,
    .long_name      = NULL_IF_CONFIG_SMALL("Adaptive Multi-Rate WideBand"),
};
