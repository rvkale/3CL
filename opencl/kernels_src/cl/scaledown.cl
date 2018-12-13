//Divide the numbers by the length after taking the fft
__kernel void
scaledown(__global float2* dataOut, __global float2* dataIn, int length, int blulength, int offset) {

    // Calculate indices
    int local_idx = get_local_id(0); // Work-item index within workgroup
    int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
    int grp_id = get_group_id(0); // Index of workgroup
    int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

    for (int i = grp_id * grp_sz + local_idx; i < blulength; i += 1) {
	float2 dd, tempip;
    tempip = dataIn[i+offset];
    dd.x = tempip.x / length;
    dd.y = tempip.y / length;
    dataOut[i+offset] = dd;
    }
}
