#define ONEPI 3.14159265359

__kernel void
vartwiddlefa(__global float2* dataOut, __global float2* dataIn, int origlength, int extenlength, int fftdirec, int offset) {
    int mul_sign;
    int local_idx = get_local_id(0);
    int grp_sz = get_local_size(0);
    int grp_id = get_local_id(0);
    int grp_offset = get_num_groups(0) * grp_sz;

    if(fftdirec == 1) {
        mul_sign = -1;
    } else {
        mul_sign = 1;
    }

    for (int i = grp_id * grp_sz + local_idx; i < extenlength; i += 1) {
        float2 dd, temp, tempip;
        float theta, tempsin, xx, yy, xy, yx;
        theta = i * i * ONEPI / origlength;
        if(i < origlength) {
            tempip = dataIn[i+offset];
            temp.x = cos(theta);
            tempsin = sin(theta);
            temp.y = mul_sign * tempsin;
            xx = tempip.x * temp.x;
            yy = tempip.y * temp.y;
            xy = tempip.x * temp.y;
            yx = tempip.y * temp.x;
            dd.x = xx - yy;
            dd.y = xy + yx;
        } else {
            dd.x = 0;
            dd.y = 0;
        }
        dataOut[i+offset] = dd;
    }
}
