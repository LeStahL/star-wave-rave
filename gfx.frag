/* Star Wave Rave by Team210 - 64k Demo at Vortex III 2k18
 * Copyright (C) 2018  Alexander Kraus <nr4@z10.info>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#version 130

// Uniforms
uniform float iTime;
uniform vec2 iResolution;
uniform sampler2D iFont;
uniform float iFontWidth;

// Global constants
const vec3 c = vec3(1.,0.,-1.);
const float pi = acos(-1.);

// Global variables
float size = 1.,
    dmin = 1.;
vec2 carriage = c.yy, 
    glyphsize = c.yy;

// Hash function
float rand(vec2 x)
{
    return fract(sin(dot(x-1. ,vec2(12.9898,78.233)))*43758.5453);
}

// 2D value noise
float valuenoise(vec2 x)
{
    vec2 y = floor(x);
    x = fract(x);
    float r00 = -1.+2.*rand(y),
        r10 = -1.+2.*rand(y+c.xy),
        r01 = -1.+2.*rand(y+c.yx),
        r11 = -1.+2.*rand(y+c.xx);
    return mix(
        mix(r00, r10, x.x),
        mix(r01, r11, x.x),
        x.y
    );
}

// Multi-frequency value noise
float mfvaluenoise(vec2 x, float f0, float f1, float phi)
{
    float sum = 0.;
    float a = 1.2;
    
    for(float f = f0; f<f1; f = f*2.)
    {
        sum = a*valuenoise(f*x) + sum;
        a = a*phi;
    }
    
    return sum;
}
    
// Add objects to scene with proper antialiasing
vec4 add(vec4 sdf, vec4 sda)
{
    return vec4(
        min(sdf.x, sda.x), 
        mix(sda.gba, sdf.gba, smoothstep(-1.5/iResolution.y, 1.5/iResolution.y, sda.x))
    );
}

// Distance to line segment
float lineseg(vec2 x, vec2 p1, vec2 p2)
{
    vec2 d = p2-p1;
    return length(x-mix(p1, p2, clamp(dot(x-p1, d)/dot(d,d),0.,1.)));
}

// Distance to stroke for any object
float stroke(float d, float w)
{
    return abs(d)-w;
}

//distance to quadratic bezier spline with parameter t
float dist(vec2 p0,vec2 p1,vec2 p2,vec2 x,float t)
{
    t = clamp(t, 0., 1.);
    return length(x-pow(1.-t,2.)*p0-2.*(1.-t)*t*p1-t*t*p2);
}

// length function, credits go to IQ / rgba; https://www.shadertoy.com/view/MdyfWc
float length2( in vec2 v  ) { return dot(v,v); }

//minimum distance to quadratic bezier spline
float spline2(vec2 p0, vec2 p1, vec2 p2, vec2 x)
{
    // check bbox, credits go to IQ / rgba; https://www.shadertoy.com/view/MdyfWc
	vec2 bmi = min(p0,min(p1,p2));
    vec2 bma = max(p0,max(p1,p2));
    vec2 bce = (bmi+bma)*0.5;
    vec2 bra = (bma-bmi)*0.5;
    float bdi = length2(max(abs(x-bce)-bra,0.0));
    if( bdi>dmin )
        return dmin;
        
    //coefficients for 0 = t^3 + a * t^2 + b * t + c
    vec2 E = x-p0, F = p2-2.*p1+p0, G = p1-p0;
    vec3 ai = vec3(3.*dot(G,F), 2.*dot(G,G)-dot(E,F), -dot(E,G))/dot(F,F);

	//discriminant and helpers
    float tau = ai.x/3., p = ai.y-tau*ai.x, q = - tau*(tau*tau+p)+ai.z, dis = q*q/4.+p*p*p/27.;
    
    //triple real root
    if(dis > 0.) 
    {
        vec2 ki = -.5*q*c.xx+sqrt(dis)*c.xz, ui = sign(ki)*pow(abs(ki), c.xx/3.);
        return dist(p0,p1,p2,x,ui.x+ui.y-tau);
    }
    
    //three distinct real roots
    float fac = sqrt(-4./3.*p), arg = acos(-.5*q*sqrt(-27./p/p/p))/3.;
    vec3 t = c.zxz*fac*cos(arg*c.xxx+c*pi/3.)-tau;
    return min(
        dist(p0,p1,p2,x, t.x),
        min(
            dist(p0,p1,p2,x,t.y),
            dist(p0,p1,p2,x,t.z)
        )
    );
}

// Read short value from texture at index off
float rshort(float off)
{
    // Parity of offset determines which byte is required.
    float hilo = mod(off, 2.);
    // Find the pixel offset your data is in (2 unsigned shorts per pixel).
    off *= .5;
    // - Determine texture coordinates.
    //     offset = i*iFontWidth+j for (i,j) in [0,iFontWidth]^2
    //     floor(offset/iFontWidth) = floor((i*iFontwidth+j)/iFontwidth)
    //                              = floor(i)+floor(j/iFontWidth) = i
    //     mod(offset, iFontWidth) = mod(i*iFontWidth + j, iFontWidth) = j
    // - For texture coordinates (i,j) has to be rescaled to [0,1].
    // - Also we need to add an extra small offset to the texture coordinate
    //   in order to always "hit" the right pixel. Pixel width is
    //     1./iFontWidth.
    //   Half of it is in the center of the pixel.
    vec2 ind = (vec2(mod(off, iFontWidth), floor(off/iFontWidth))+.05)/iFontWidth;
    // Get 4 bytes of data from the texture
    vec4 block = texture(iFont, ind);
    // Select the appropriate word
    vec2 data = mix(block.rg, block.ba, hilo);
    // Convert bytes to unsigned short. The lower bytes operate on 255,
    // the higher bytes operate on 65280, which is the maximum range 
    // of 65535 minus the lower 255.
    return round(dot(vec2(255., 65280.), data));
}

// Compute distance to glyph from ascii value out of the font texture.
// This function parses glyph point and control data and computes the correct
// Spline control points. Then it uses the signed distance function to
// piecewise bezier splines to get a signed distance to the font glyph.
float dglyph(vec2 x, int ascii)
{
    // Treat spaces
    if(ascii == 32)
    {
        glyphsize = size*vec2(.02,1.);
        return 1.;
    }

    // Get glyph index length
    float nchars = rshort(0.);
    
    // Find character in glyph index
    float off = -1.;
    for(float i=0.; i<nchars; i+=1.)
    {
        int ord = int(rshort(1.+2.*i));
        if(ord == ascii)
        {
            off = rshort(1.+2.*i+1);
            break;
        }
    }
    // Ignore characters that are not present in the glyph index.
    if(off == -1.) return 1.;
    
    // Get short range offsets. Sign is read separately.
    vec2 dx = mix(c.xx,c.zz,vec2(rshort(off), rshort(off+2.)))*vec2(rshort(off+1.), rshort(off+3.));
    
    // Read the glyph splines from the texture
    float npts = rshort(off+4.),
        xoff = off+5., 
        yoff = off+6.+npts,
        toff = off+7.+2.*npts, 
        coff = off+8.+3.*npts,
        ncont = rshort(coff-1.),
        d = 1.;
    
    // Save glyph size
    vec2 mx = -100.*c.xx,
        mn = 100.*c.xx;
    
    // Loop through the contours of the glyph. All of them are closed.
    for(float i=0.; i<ncont; i+=1.)
    {
        // Get the contour start and end indices from the contour array.
        float istart = 0., 
            iend = rshort(coff+i);
        if(i>0.)
            istart = rshort(coff+i-1.) + 1.;
        
        // Prepare a stack
        vec2 stack[3];
        float tstack[3];
        int stacksize = 0;
        
        // Loop through the segments
        for(float j = istart; j <= iend; j += 1.)
        {
            tstack[stacksize] = rshort(toff + j);
            stack[stacksize] = (vec2(rshort(xoff+j), rshort(yoff+j)) + dx)/65536.*size;
            mx = max(mx, stack[stacksize]);
            mn = min(mn, stack[stacksize]);
            ++stacksize;
            
            // Check if line segment is finished
            if(stacksize == 2)
            {
                if(tstack[0]*tstack[1] == 1)
                {
                    d = min(d, lineseg(x, stack[0], stack[1]));
                    --j;
                    stacksize = 0;
                }
            }
            else 
            if(stacksize == 3)
            {
                if(tstack[0]*tstack[2] == 1.)
                {
                    d = min(d, spline2(stack[0], stack[1], stack[2], x));
                    --j;
                    stacksize = 0;
                }
                else
                {
                    vec2 p = mix(stack[1], stack[2], .5);
                    d = min(d, spline2(stack[0], stack[1], p, x));
                    stack[0] = p;
                    tstack[0] = 1.;
                    mx = max(mx, stack[0]);
                    mn = min(mn, stack[0]);
                    --j;
                    stacksize = 1;
                }
            }
        }
        tstack[stacksize] = rshort(toff + istart);
        stack[stacksize] = (vec2(rshort(xoff+istart), rshort(yoff+istart)) + dx)/65536.*size;
        mx = max(mx, stack[0]);
        mn = min(mn, stack[0]); 
        ++stacksize;
        if(stacksize == 2)
        {
            d = min(d, lineseg(x, stack[0], stack[1]));
        }
        else 
        if(stacksize == 3)
        {
            d = min(d, (spline2(stack[0], stack[1], stack[2], x)));
        }
    }
    
    glyphsize = abs(mx-mn);
    
    return d;
}

// Compute distance to glyph control points for debug purposes
float dglyphpts(vec2 x, int ascii)
{
    // Get glyph index length
    float nchars = rshort(0.);
    
    // Find character in glyph index
    float off = -1.;
    for(float i=0.; i<nchars; i+=1.)
    {
        int ord = int(rshort(1.+2.*i));
        if(ord == ascii)
        {
            off = rshort(1.+2.*i+1);
            break;
        }
    }
    // Ignore characters that are not present in the glyph index.
    if(off == -1.) return 1.;
    
    // Get short range offsets. Sign is read separately.
    vec2 dx = mix(c.xx,c.zz,vec2(rshort(off), rshort(off+2.)))*vec2(rshort(off+1.), rshort(off+3.));
    
    // Read the glyph splines from the texture
    float npts = rshort(off+4.),
        xoff = off+5., 
        yoff = off+6.+npts,
        d = 1.;
        
    // Debug output of the spline control points
    for(float i=0.; i<npts; i+=1.)
    {
        vec2 xa = ( vec2(rshort(xoff+i), rshort(yoff+i)) + dx )/65536.*size;
        d = min(d, length(x-xa)-3.e-3);
    }
    
    return d;
}

// Two-dimensional rotation matrix
mat2 rot(float t)
{
    vec2 sc = vec2(cos(t), sin(t));
    return mat2(sc*c.xz, sc.yx);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord/iResolution.yy-.5;
//     vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4)); TODO: remove
    vec3 col = c.yyy;
    
    // Scene 1: 2D; Greetings and vortex logo.
    if(iTime < 1000.)
    {
        vec4 sdf = vec4(1., col);
        float d = 1., dc = 1.;
        
        vec2 vn = 2.e-2*vec2(valuenoise(2.3*uv-2.*vec2(1.5,2.4)*iTime), valuenoise(1.7*uv-2.4*vec2(1.2,2.1)*iTime));
        
        // "Hello, Vortex III"
        {
            size = 1.54;
            carriage = -.25*c.xy;
            int str[17] = int[17](72, 101, 108, 108, 111, 44, 32, 86, 111, 114, 116, 101, 120, 32, 73, 73, 73);
            for(int i=0; i<17; ++i)
            {
                d = min(d, dglyph(uv-carriage-vn, str[i]));
                dc = min(dc, dglyphpts(uv-carriage-vn, str[i]));
                carriage += glyphsize.x*c.xy + .01*c.xy;
            }
        }
        d = stroke(d, 2.e-3)+.1*length(vn);
        sdf = add(sdf, vec4(d, c.xxx));
        sdf = add(sdf, vec4(dc, c.xyy));
        
        col = sdf.gba * smoothstep(1.5/iResolution.y, -1.5/iResolution.y, sdf.x);        
    }
    
    // Set the fragment color
    fragColor = vec4(col, 1.);
}

void main()
{
    mainImage(gl_FragColor, gl_FragCoord.xy);
}
