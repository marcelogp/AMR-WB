/*
 * Alpha optimized DSP utils
 * Copyright (c) 2002 Falk Hueffner <falk@debian.org>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include "asm.h"
#include "../dsputil.h"
#include "../mpegvideo.h"

extern UINT8 zigzag_end[64];

static void dct_unquantize_h263_mvi(MpegEncContext *s, DCTELEM *block,
                                    int n, int qscale)
{
    int i, n_coeffs;
    uint64_t qmul, qadd;
    uint64_t correction;
    DCTELEM *orig_block = block;
    DCTELEM block0;

    ASM_ACCEPT_MVI;

    if (s->mb_intra) {
        if (!s->h263_aic) {
            if (n < 4) 
                block0 = block[0] * s->y_dc_scale;
            else
                block0 = block[0] * s->c_dc_scale;
        }
        n_coeffs = 64; // does not always use zigzag table 
    } else {
        n_coeffs = zigzag_end[s->block_last_index[n]];
    }

    qmul = qscale << 1;
    qadd = WORD_VEC((qscale - 1) | 1);
    /* This mask kills spill from negative subwords to the next subword.  */ 
    correction = WORD_VEC((qmul - 1) + 1); /* multiplication / addition */

    for(i = 0; i < n_coeffs; block += 4, i += 4) {
        uint64_t levels, negmask, zeros, add;

        levels = ldq(block);
        if (levels == 0)
            continue;

        negmask = maxsw4(levels, -1); /* negative -> ffff (-1) */
        negmask = minsw4(negmask, 0); /* positive -> 0000 (0) */

        zeros = cmpbge(0, levels);
        zeros &= zeros >> 1;
        /* zeros |= zeros << 1 is not needed since qadd <= 255, so
           zapping the lower byte suffices.  */

        levels *= qmul;
        levels -= correction & (negmask << 16);

        /* Negate qadd for negative levels.  */
        add = qadd ^ negmask;
        add += WORD_VEC(0x0001) & negmask;
        /* Set qadd to 0 for levels == 0.  */
        add = zap(add, zeros);

        levels += add;

        stq(levels, block);
    }

    if (s->mb_intra && !s->h263_aic)
        orig_block[0] = block0;
}

void MPV_common_init_axp(MpegEncContext *s)
{
    if (amask(AMASK_MVI) == 0) {
        s->dct_unquantize_h263 = dct_unquantize_h263_mvi;
    }
}
