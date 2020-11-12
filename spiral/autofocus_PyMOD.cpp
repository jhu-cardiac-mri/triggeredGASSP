// Author: Mike Schär
// Date: 2015Oct14

#include "PyFI/PyFI.h"
using namespace PyFI; /* for PyFI::Array */

#include <iostream>

PYFI_FUNC(blur)
{
    PYFI_START();

    /* Required */
    PYFI_POSARG(Array<complex<float> >, data);
    PYFI_POSARG(Array<complex<float> >, rot_matrix);

    /* OPTIONAL */
    //PYFI_KWARG(float, adc, 15.);

    /* copy an input array size for output */
    PYFI_SETOUTPUT_ALLOC(Array<complex<float> >, out_data, rot_matrix->dims_object());
    
    /* parameters depending on data */
    int64_t nr_x    = rot_matrix->dimensions(0);
    int64_t nr_y    = rot_matrix->dimensions(1);
    int64_t nr_freq = 1;
    int64_t multiple_frequences = 0;
    if (rot_matrix->ndim() == 3)
    {
        multiple_frequences = 1;
        nr_freq = rot_matrix->dimensions(2);
    }
    //float a,b,c,d;
    Array<complex<float> > temp_array(*data);
    
    /* PERFORM */
    FFTW::fft2(*data, *data, FFTW_BACKWARD);
    
    if (multiple_frequences)
    {
        for (int freq=0; freq<nr_freq; freq++)
        {
            for (int x=0; x<nr_x; x++)
            {
                for (int y=0; y<nr_y; y++)
                {
                    temp_array(x,y) = (*data)(x,y) * (*rot_matrix)(x,y,freq);
                }
            }
            FFTW::fft2(temp_array, temp_array, FFTW_FORWARD);
            for (int x=0; x<nr_x; x++)
            {
                for (int y=0; y<nr_y; y++)
                {
                    (*out_data)(x,y,freq) = temp_array(x,y);
                }
            }
        }
    } else
    {
        for (int x=0; x<nr_x; x++)
        {
            for (int y=0; y<nr_y; y++)
            {
                (*out_data)(x,y) = (*data)(x,y) * (*rot_matrix)(x,y);
            }
        }
        FFTW::fft2(*out_data, *out_data, FFTW_FORWARD);
    }

    PYFI_END();
} /* end blur() */


PYFI_FUNC(blur3D)
{
    PYFI_START();
    
    /* Required */
    PYFI_POSARG(Array<complex<float> >, data);
    PYFI_POSARG(Array<complex<float> >, rot_matrix);
    
    /* OPTIONAL */
    //PYFI_KWARG(float, adc, 15.);
    
    /* copy an input array size for output */
    PYFI_SETOUTPUT_ALLOC(Array<complex<float> >, out_data, rot_matrix->dims_object());
    
    /* parameters depending on data */
    int64_t nr_x    = rot_matrix->dimensions(0);
    int64_t nr_y    = rot_matrix->dimensions(1);
    int64_t nr_z    = rot_matrix->dimensions(2);
    int64_t nr_freq = 1;
    int64_t multiple_frequences = 0;
    if (rot_matrix->ndim() == 4)
    {
        multiple_frequences = 1;
        nr_freq = rot_matrix->dimensions(3);
    }
    //float a,b,c,d;
    Array<complex<float> > temp_array(*data);
    
    /* PERFORM */
    FFTW::fft2(*data, *data, FFTW_BACKWARD);

    if (multiple_frequences)
    {
        for (int freq=0; freq<nr_freq; freq++)
        {
            for (int x=0; x<nr_x; x++)
            {
                for (int y=0; y<nr_y; y++)
                {
                    for (int z=0; z<nr_z; z++)
                    {
                        temp_array(x,y,z) = (*data)(x,y,z) * (*rot_matrix)(x,y,z,freq);
                    }
                }
            }
            FFTW::fft2(temp_array, temp_array, FFTW_FORWARD);
            for (int x=0; x<nr_x; x++)
            {
                for (int y=0; y<nr_y; y++)
                {
                    for (int z=0; z<nr_z; z++)
                    {
                        (*out_data)(x,y,z,freq) = temp_array(x,y,z);
                    }
                }
            }
        }
    } else
    {
        for (int x=0; x<nr_x; x++)
        {
            for (int y=0; y<nr_y; y++)
            {
                for (int z=0; z<nr_z; z++)
                {
                    (*out_data)(x,y,z) = (*data)(x,y,z) * (*rot_matrix)(x,y,z);
                }
            }
        }
        FFTW::fft2(*out_data, *out_data, FFTW_FORWARD);
    }
    
    PYFI_END();
} /* end blur3D() */


PYFI_FUNC(zerofill)
{
    PYFI_START();
    
    /* Required */
    PYFI_POSARG(Array<complex<float> >, data);
    PYFI_POSARG(Array<int64_t>, outdim);
    
    /***** ALLOCATE OUTPUT */
    PYFI_SETOUTPUT_ALLOC(Array<complex <float> >, out, DA(*outdim));
    
    /* PERFORM */
    FFTW::fft2(*data, *data, FFTW_BACKWARD);
    
    // copy inverse Fourier transformed input data into output Array in centered manner
    *out = (complex<float>) 0;
    out->insert(*data);
    FFTW::fft2(*out, *out, FFTW_FORWARD);
    
    PYFI_END();
} /* end zf() */


PYFI_FUNC(filtered_zerofill)
{
    PYFI_START();
    
    /* Required */
    PYFI_POSARG(Array<complex<float> >, data);
    PYFI_POSARG(Array<float>, filter);
    PYFI_POSARG(Array<int64_t>, outdim);
    
    /***** ALLOCATE OUTPUT */
    PYFI_SETOUTPUT_ALLOC(Array<complex <float> >, out, DA(*outdim));
    
    /* PERFORM */
    FFTW::fft2(*data, *data, FFTW_BACKWARD);
    
    /* parameters depending on data */
    int64_t nr_x    = data->dimensions(0);
    int64_t nr_y    = data->dimensions(1);
    for (int x=0; x<nr_x; x++)
    {
        for (int y=0; y<nr_y; y++)
        {
            (*data)(x,y) = (*data)(x,y) * (*filter)(x,y);
        }
    }
    
    // copy inverse Fourier transformed input data into output Array in centered manner
    *out = (complex<float>) 0;
    out->insert(*data);
    FFTW::fft2(*out, *out, FFTW_FORWARD);
    
    PYFI_END();
} /* end zf() */

/* ##############################################################
 *                  MODULE DESCRIPTION
 * ############################################################## */

/* list of functions to be accessible from python
 */
PYFI_LIST_START_
    PYFI_DESC(blur, "out = blur()\n blur image for different off-resonance frequences.")
    PYFI_DESC(blur3D, "out = blur()\n blur image for different off-resonance frequences.")
    PYFI_DESC(zerofill, "out = zerofill()\n zero-fill image at userdefined factor.")
    PYFI_DESC(filtered_zerofill, "out = filtered_zerofill()\n filter and zero-fill image at userdefined factor.")
PYFI_LIST_END_

