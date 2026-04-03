/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <stdint.h>

#include "RISBufferSegmentParameters.h"

namespace rtxdi
{

class RISBufferSegmentAllocator
{
public:
    RISBufferSegmentAllocator();
    // Returns starting offset of segment in buffer
    uint32_t allocateSegment(uint32_t sizeInElements);
    uint32_t getTotalSizeInElements() const;

private:
    uint32_t m_totalSizeInElements;
};

}
