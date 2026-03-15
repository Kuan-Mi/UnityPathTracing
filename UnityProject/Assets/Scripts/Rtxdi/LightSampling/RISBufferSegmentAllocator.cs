// Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

namespace Rtxdi.LightSampling
{
    public class RISBufferSegmentAllocator
    {
        private uint m_totalSizeInElements;

        public RISBufferSegmentAllocator()
        {
            m_totalSizeInElements = 0;
        }

        /// <summary>
        /// Returns starting offset of segment in buffer.
        /// </summary>
        public uint AllocateSegment(uint sizeInElements)
        {
            uint prevSize = m_totalSizeInElements;
            m_totalSizeInElements += sizeInElements;
            return prevSize;
        }

        public uint GetTotalSizeInElements()
        {
            return m_totalSizeInElements;
        }
    }
}
