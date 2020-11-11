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
 */

#ifndef KAISERBESSEL_KERNEL_GUARD
#define KAISERBESSEL_KERNEL_GUARD
/* Ken Johnson and Nick Zwart
 *
 * Kernel implementation from:
 *   Beatty, Philip J., Dwight G. Nishimura, and John M. Pauly. "Rapid gridding
 *   reconstruction with a minimal oversampling ratio." Medical Imaging, IEEE
 *   Transactions on 24.6 (2005): 799-808.
 */

#define KERNEL_WIDTH               5.0  // in k-space pixels
#define DEFAULT_RADIUS_FOV_PRODUCT ((KERNEL_WIDTH) / 2.0)

#define sqr(__se)          ((__se)*(__se))

/*------------------------------------------------------------*/
/* PURPOSE: Evaluate modified Bessel function In(x) and n=0.  */
/*------------------------------------------------------------*/
static double i0(double x)
{
    double ax,ans;
    double y;


    if ((ax=fabs(x)) < 3.75)
    {
        y=x/3.75,y=y*y;
        ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
                                             +y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
    }
    else
    {
        y=3.75/ax;
        ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1
                                              +y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
                                                      +y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
                                                              +y*0.392377e-2))))))));
    }
    return ans;
}

static double kernel(double radius, double beta)
{
    return (i0 (beta * sqrt (1 - sqr(radius))) / (i0(beta)));
}

void _kaiserbessel(Array<float> &kernel_table_out, double &oversampling_ratio)
{
    int i;
    int size = kernel_table_out.dimensions(0);
    double beta = (M_PI*sqrt(sqr(KERNEL_WIDTH/oversampling_ratio*(oversampling_ratio-0.5))-0.8));
    assert(size > 0);

    for (i=1; i<size-1; i++)
    {
        get1(kernel_table_out, i) = kernel( sqrt(i/(double)(size-1)), beta); // kernel table for radius squared
        assert(!isnan(get1(kernel_table_out, i)));
    }
    get1(kernel_table_out, 0) = 1.0;
    get1(kernel_table_out, size-1) = 0.0;

    assert(fabs(get1(kernel_table_out, 0) - __kernel(0.00000000001)) <= 0.00001);
}

#endif // GUARD
