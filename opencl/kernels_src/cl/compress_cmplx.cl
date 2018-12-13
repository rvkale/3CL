// Kernel to transfer an array of reals to a complex array
__kernel void
compress_cmplx(__global float* dst, __global float2* src, const unsigned int count, const unsigned int iOffset, const unsigned int oOffset)
{
	// Calculate indices
	int local_idx = get_local_id(0); // Work-item index within workgroup
	int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
	int grp_id = get_group_id(0); // Index of workgroup
	int global_idx = grp_id * grp_sz + local_idx; // Calculate global index of work-item
	int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

	while(global_idx < count) {
		float2 a0 = src[global_idx + iOffset];
		dst[global_idx + oOffset] = a0.x;
		global_idx += grp_offset;
	}
}
