/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan
 * PSL v2. You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PSL v2 for more details.
 * ------------------------------------------------------------------------- */

#ifndef ROW_COL_INT8_QUANT_TILING_H
#define ROW_COL_INT8_QUANT_TILING_H
#include <cstdint>

struct RowColInt8QuantTilingData {
  uint32_t m;
  uint32_t n;
  uint32_t rowTiles;
  uint32_t colTiles;
  uint32_t blocks;
};

#endif  // ROW_COL_INT8_QUANT_TILING_H
