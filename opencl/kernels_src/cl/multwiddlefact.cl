#define ONEPI 3.14159265359
// Tranpose of complex matrix
// Offset is used to specify where the first row, first column entry of the matrix
// is located in linear memory space
__kernel void
multwiddlefact(__global float2* dataOut, int origlength, int extenlength, int fftdirec, int offset) {

    int mul_sign;
    // width = N (signal length)
	// height = batch_size (number of signals in a batch)
    int local_idx = get_local_id(0); // Work-item index within workgroup
    int grp_sz = get_local_size(0); // Total number of work-items in each workgroup
    int grp_id = get_group_id(0); // Index of workgroup
    int grp_offset = get_num_groups(0) * grp_sz; // Offset for memory access

    if (fftdirec == 1) {
        mul_sign = 1;
    } else {
        mul_sign = -1;
    }
    //for (int i = grp_id * grp_sz + local_idx; i < extenlength; i += grp_offset) {
    for (int i = grp_id * grp_sz + local_idx; i < extenlength; i += 1) {
        float2 dd;
	    float theta;
        float tempsin;

        if (i < origlength) {
            theta = i * i * ONEPI / origlength;
            dd.x = cos(theta);
            tempsin = sin(theta);
            dd.y = mul_sign * tempsin;
        } else if (i >(extenlength - origlength) && i < extenlength) {
            theta = (i - extenlength) * (i - extenlength) * ONEPI / origlength;
            dd.x = cos(theta);
            tempsin = sin(theta);
            dd.y = mul_sign * tempsin;
        } else {
            dd.x = 0;
            dd.y = 0;
        }

	    
	    dataOut[i+offset] = dd;
    }
}
