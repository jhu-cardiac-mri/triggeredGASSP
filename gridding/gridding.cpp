/* Copyright (c) 2014, Dignity Health
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Nick Zwart, Ken Johnson
 * Date: 2014jul15
 * Convolution based Gridding, Rolloff (Deapodization), and De-Gridding.
 */

#ifndef GRIDDING_CPP_GUARD
#define GRIDDING_CPP_GUARD

#include "triggeredGASSP/gridding/kaiserbessel.cpp"

/* return forced template */
#ifndef ITSELF_FUNC
#define ITSELF_FUNC
template <class T>
inline T itself(T arg)
{
    return arg;
}
#endif

#define _sqr(__se) ((__se)*(__se))
#define dist2(_x,_y) (_sqr(_x)+_sqr(_y))

/* check the current grid coordinate against limits */
template<class T>
void set_minmax(T x, int *min, int *max, int maximum, T radius)
{
    *min = (int) ceil(x - radius);
    *max = (int) floor(x + radius);
    if (*min < 0)
        *min = 0;
    if (*max >= maximum)
        *max = maximum-1;
}

/* GRID
 *  data: nD array with 1-vec dimensions equal to coords array.
 *  coords: nD array with 2-vec.
 *  weights: nD array with 1-vec (dims equal to coords array).  This holds the
 *           density compensation for each gridded point.
 *  kernel_table: 1D array with Kaiser-Bessel kernel table
 *  out: 2D array with equal dimensions (m == n).
 *  dx, dy: scaler pixel shift in
 */
template<class T>
void _grid2_thread(int *num_threads, int *cur_thread, Array<complex<T> > &data, Array<T> &coords, Array<T> &weight, Array<complex<T> > &out, Array<T> &kernel_table, T dx, T dy)
{
    int imin, imax, jmin, jmax, i, j;
    int width = out.dimensions(0); // assume isotropic dims
    int width_div2 = width / 2;
    uint64_t arms = 1;
    uint64_t points_in_arm = 1;
    uint64_t arms_times_coils = 1;
    uint64_t coils = 1;
    uint64_t coil, arm, point;
    uint64_t width_times_coil = width;
    T x, y, ix, jy;
    T kernelRadius = DEFAULT_RADIUS_FOV_PRODUCT / width;
    T kernelRadius_sqr = kernelRadius * kernelRadius;
    T width_inv = 1.0 / width;
    
    T dist_multiplier = (kernel_table.dimensions(0) - 1)/kernelRadius_sqr;
    
    out = complex<T>(0.0);
    
    if (coords.ndim() == 3)
    {
        points_in_arm = coords.dimensions(1);
        arms = coords.dimensions(2);
    }
    
    if (out.ndim() == 3)
    {
        coils = out.dimensions(2);
    }
    arms_times_coils = arms * coils;
    width_times_coil = width * coils;
    
    /* split threads up by even chunks of data_out memory */
    uint64_t numSections = coils / *num_threads;
    uint64_t coil_start = *cur_thread * numSections;
    uint64_t coil_end = (*cur_thread+1) * numSections;
    
    /* loop over output data points */
    for (coil=coil_start; coil<coil_end; coil++)
    {
        for (arm=0; arm<arms; arm++)
        {
            for (point=0; point<points_in_arm; point++)
            {
                complex<T> d = data(point,arm,coil) * weight(point,arm);
                
                /* get the coordinates of the datapoint to grid
                 *  these vary between -.5 -- +.5               */
                x = coords(0,point,arm);
                y = coords(1,point,arm);
                
                /* add shift phase */
                d *= exp( complex<T>(0, -2.*M_PI*(x*dx+y*dy)) );
                
                /* set the boundaries of final dataset for gridding this point */
                ix = x * width + width_div2;
                set_minmax(ix, &imin, &imax, width, (T) DEFAULT_RADIUS_FOV_PRODUCT);
                jy = y * width + width_div2;
                set_minmax(jy, &jmin, &jmax, width, (T) DEFAULT_RADIUS_FOV_PRODUCT);
                
                /* grid this point onto the neighboring cartesian points */
                for (j=jmin; j<=jmax; ++j)
                {
                    jy = (j - width_div2) * width_inv;
                    for (i=imin; i<=imax; ++i)
                    {
                        ix = (i - width_div2) * width_inv;
                        T dist_sqr = dist2(ix - x, jy - y);
                        if (dist_sqr < kernelRadius_sqr)
                        {
                            T ker = get1(kernel_table, (int) rint(dist_sqr * dist_multiplier));
                            out(i,j,coil) += ker * d;
                        }
                    }// x
                }// y
            }//point in arms
        } //arms
    } //coil
}

/* the actual calling function for a threaded grid */
template <class T>
void _grid2_threaded(Array<complex<T> > *data, Array<T> *crds, Array<T> *weight, Array<complex<T> > *outdata, Array<T> *kernel, T dx, T dy, int num_threads)

{
    /* Make sure output array is initialized */
    *outdata = complex<T>(0.0);
    
    /* start threads */
    create_threads7(num_threads,
                    itself(_grid2_thread<T>),
                    data,
                    crds,
                    weight,
                    outdata,
                    kernel,
                    dx,
                    dy);
    
} // end grid2_threaded()

/* GRID
 *  data: nD array with 1-vec dimensions equal to coords array.
 *  coords: nD array with 2-vec.
 *  weights: nD array with 1-vec (dims equal to coords array).  This holds the
 *           density compensation for each gridded point.
 *  kernel_table: 1D array with Kaiser-Bessel kernel table
 *  out: 2D array with equal dimensions (m == n).
 *  dx, dy: scaler pixel shift in
 */
template<class T>
void _grid2(Array<complex<T> > &data, Array<T> &coords, Array<T> &weight, Array<complex<T> > &out, Array<T> &kernel_table, T dx, T dy)
{
    int imin, imax, jmin, jmax, i, j;
    int width = out.dimensions(0); // assume isotropic dims
    int width_div2 = width / 2;
    uint64_t arms = 1;
    uint64_t points_in_arm = 1;
    uint64_t arms_times_coils = 1;
    uint64_t coils = 1;
    uint64_t coil, arm, point;
    uint64_t width_times_coil = width;
    T x, y, ix, jy;
    T kernelRadius = DEFAULT_RADIUS_FOV_PRODUCT / width;
    T kernelRadius_sqr = kernelRadius * kernelRadius;
    T width_inv = 1.0 / width;

    T dist_multiplier = (kernel_table.dimensions(0) - 1)/kernelRadius_sqr;

    out = complex<T>(0.0);
    
    if (coords.ndim() == 3)
    {
        points_in_arm = coords.dimensions(1);
        arms = coords.dimensions(2);
    }
    
    if (out.ndim() == 3)
    {
        coils = out.dimensions(2);
    }
    arms_times_coils = arms * coils;
    width_times_coil = width * coils;
    
    /* loop over output data points */
    for (coil=0; coil<coils; coil++)
    {
        for (arm=0; arm<arms; arm++)
        {
            for (point=0; point<points_in_arm; point++)
            {
                complex<T> d = data(point,arm,coil) * weight(point,arm);
                
                /* get the coordinates of the datapoint to grid
                 *  these vary between -.5 -- +.5               */
                x = coords(0,point,arm);
                y = coords(1,point,arm);
                
                /* add shift phase */
                d *= exp( complex<T>(0, -2.*M_PI*(x*dx+y*dy)) );
                
                /* set the boundaries of final dataset for gridding this point */
                ix = x * width + width_div2;
                set_minmax(ix, &imin, &imax, width, (T) DEFAULT_RADIUS_FOV_PRODUCT);
                jy = y * width + width_div2;
                set_minmax(jy, &jmin, &jmax, width, (T) DEFAULT_RADIUS_FOV_PRODUCT);
                
                /* grid this point onto the neighboring cartesian points */
                for (j=jmin; j<=jmax; ++j)
                {
                    jy = (j - width_div2) * width_inv;
                    for (i=imin; i<=imax; ++i)
                    {
                        ix = (i - width_div2) * width_inv;
                        T dist_sqr = dist2(ix - x, jy - y);
                        if (dist_sqr < kernelRadius_sqr)
                        {
                            T ker = get1(kernel_table, (int) rint(dist_sqr * dist_multiplier));
                            out(i,j,coil) += ker * d;
                        }
                    }// x
                }// y
            }//point in arms
        } //arms
    } //coil
}

/* DEGRID
 *  data: 2D array with equal dimensions (m == n).
 *  coords: nD array with 2-vec.
 *  out: nD array with 1-vec dimensions equal to coords array.
 *  kernel_table: 1D array with Kaiser-Bessel kernel table
 */

/*
 * The work done by an individual thread
 *    -this could be called with a macro for
 *     1 thread, and 0 current thread and it
 *     should behave the same as _degrid2()
 */
template<class T>
void _degrid2_thread(int *num_threads, int *cur_thread, Array<complex<T> > &data, Array<T> &coords, Array<complex<T> > &out, Array<T> &kernel_table)
{
    int imin, imax, jmin, jmax, i, j;
    int width = data.dimensions(0); // assume isotropic dims
    int width_div2 = width / 2;
    uint64_t p;
    T x, y, ix, jy;
    T kernelRadius = DEFAULT_RADIUS_FOV_PRODUCT / width;
    T kernelRadius_sqr = kernelRadius * kernelRadius;
    T width_inv = 1.0 / width;
    
    T dist_multiplier = (kernel_table.dimensions(0) - 1)/kernelRadius_sqr;
    
    /* split threads up by even chunks of data_out memory */
    unsigned long numSections = out.size() / *num_threads;
    unsigned long p_start = *cur_thread * numSections;
    unsigned long p_end = (*cur_thread+1) * numSections;
    
    /* loop over output data points */
    for (p=p_start; p<p_end; p++)
    {
        complex<T> d = 0.;
        
        /* get the coordinates of the datapoint to grid
         *  these vary between -.5 -- +.5               */
        x = coords.get1v(p, 0);
        y = coords.get1v(p, 1);
        
        /* set the boundaries of final dataset for gridding this point */
        ix = x * width + width_div2;
        set_minmax(ix, &imin, &imax, width, (T) DEFAULT_RADIUS_FOV_PRODUCT);
        jy = y * width + width_div2;
        set_minmax(jy, &jmin, &jmax, width, (T) DEFAULT_RADIUS_FOV_PRODUCT);
        
        /* Convolve the kernel at the coordinate location to get a
         * non-cartesian sample */
        for (j=jmin; j<=jmax; ++j)
        {
            jy = (j - width_div2) * width_inv;
            for (i=imin; i<=imax; ++i)
            {
                ix = (i - width_div2) * width_inv;
                T dist_sqr = dist2(ix - x, jy - y);
                if (dist_sqr < kernelRadius_sqr)
                {
                    T ker = get1(kernel_table, (int) rint(dist_sqr * dist_multiplier));
                    d += get2(data, i, j) * ker; // convolution sum
                }
            }
        }
        
        get1(out, p) = d; // store the sum for this coordinate point
    }
}




/* the actual calling function for a threaded de-grid */
template <class T>
void _degrid2_threaded(Array<complex<T> > *data,
                      Array<T> *crds,
                      Array<complex<T> > *outdata,
                      Array<T> *kernel,
                      int num_threads)

{
    /* Make sure output array is initialized */
    *outdata = complex<T>(0.0);
    
    /* start threads */
    create_threads4(num_threads,
                    itself(_degrid2_thread<T>),
                    data,
                    crds,
                    outdata,
                    kernel);
    
} // end degrid2_threaded()

template<class T>
void _degrid2(Array<complex<T> > &data, Array<T> &coords, Array<complex<T> > &out, Array<T> &kernel_table)
{
    int imin, imax, jmin, jmax, i, j;
    int width = data.dimensions(0); // assume isotropic dims
    int width_div2 = width / 2;
    uint64_t p;
    T x, y, ix, jy;
    T kernelRadius = DEFAULT_RADIUS_FOV_PRODUCT / width;
    T kernelRadius_sqr = kernelRadius * kernelRadius;
    T width_inv = 1.0 / width;

    T dist_multiplier = (kernel_table.dimensions(0) - 1)/kernelRadius_sqr;

    out = complex<T>(0.0);

    /* loop over output data points */
    for (p=0; p<out.size(); p++)
    {
        complex<T> d = 0.;

        /* get the coordinates of the datapoint to grid
         *  these vary between -.5 -- +.5               */
        x = coords.get1v(p, 0);  
        y = coords.get1v(p, 1);  

        /* set the boundaries of final dataset for gridding this point */
        ix = x * width + width_div2;
        set_minmax(ix, &imin, &imax, width, (T) DEFAULT_RADIUS_FOV_PRODUCT);
        jy = y * width + width_div2;
        set_minmax(jy, &jmin, &jmax, width, (T) DEFAULT_RADIUS_FOV_PRODUCT);

        /* Convolve the kernel at the coordinate location to get a
         * non-cartesian sample */
        for (j=jmin; j<=jmax; ++j)
        {
            jy = (j - width_div2) * width_inv;
            for (i=imin; i<=imax; ++i)
            {
                ix = (i - width_div2) * width_inv;
                T dist_sqr = dist2(ix - x, jy - y);
                if (dist_sqr < kernelRadius_sqr)
                {
                    T ker = get1(kernel_table, (int) rint(dist_sqr * dist_multiplier));
                    d += get2(data, i, j) * ker; // convolution sum
                }
            }
        }

        get1(out, p) = d; // store the sum for this coordinate point
    }
}

/* FOV CROP
 * Outputs a the input image multiplied by a 2D circular mask.  The diameter of
 * the circle is the length of the first dim of the input array
 * (assumes m == n).
 */
template<class T>
void crop_circle (Array<complex<T> > &in)	
{
	int64_t size = in.dimensions(0);
	int64_t r2 = size * size / 4;
	for(int64_t j=0; j<size; j++) 
    {
		int64_t y = j - size/2;
		for(int64_t i=0; i<size; i++) 
        {
			int64_t x = i - size/2;
			if (x*x + y*y > r2)	
            {
				get2(in, i, j) = get2(in, i, j) = 0.0;
			}
		}
	}
}

/* ROLLOFF
 * Deapodize by sampling 2D grid kernel.
 * in: 2D array (m == n)
 *  kernel_table: 1D array with Kaiser-Bessel kernel table
 * out: 2D array
 */
template<class T>
void _rolloff2(Array<complex<T> > &in, Array<complex<T> > &out, Array<T> &kernel_table, int32_t cropfilt)
{
    /* get grid dimensionality for scaling */
    int64_t gridMtx = in.dimensions(0);
    int64_t effMtx = out.dimensions(0);
    T osf = (T) gridMtx / (T) effMtx;
    osf *= osf;

    /* delta function at k0 to sample the grid kernel */
    Array<T> delta_crd(2);
    Array<T> delta_wgt(1);
    delta_wgt(0) = (T) 1;
    Array<complex<T> > delta_dat(1);
    delta_dat(0) = complex<T>(1,0);

    /* create another grid the same size as the oversampled grid 
     * to hold the deapodization filter. */
    Array<complex<T> > rolloff(in.dimensions_vector());
    _grid2(delta_dat, delta_crd, delta_wgt, rolloff, kernel_table, (T) 0., (T) 0.);
    fft2(rolloff, rolloff, FFTW_FORWARD);

    /* take magnitude of each element and divide */
    for (uint64_t i=0; i<in.size(); ++i)
    {
        T mag = abs(rolloff(i));
        if (mag > 0.)
            rolloff(i) = in(i)/mag * osf;
        else
            rolloff(i) = complex<T>(0.);
    }

    /* reduce the grid size */
    out.insert(rolloff);

    /* apply circle filter */
    if (cropfilt) crop_circle(out);
}

#endif // GUARD
