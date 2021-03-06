/* File generated with Shader Minifier 1.1.5
 * http://www.ctrl-alt-test.fr
 */
#ifndef GFX_H_
# define GFX_H_

const char *gfx_frag =
 "#version 130\n"
 "uniform float iTime;"
 "uniform vec2 iResolution;"
 "uniform sampler2D iFont;"
 "uniform float iFontWidth,iNBeats,iScale;"
 "const vec3 c=vec3(1.,0.,-1.);"
 "const float pi=acos(-1.);"
 "float size=1.,dmin=1.;"
 "vec2 carriage=c.yy,glyphsize=c.yy;"
 "vec3 col=c.yyy;"
 "float rand(vec2 x)"
 "{"
   "return fract(sin(dot(x-1.,vec2(12.9898,78.233)))*43758.5);"
 "}"
 "float rand(vec3 x)"
 "{"
   "return fract(sin(dot(x-1.,vec3(12.9898,78.233,33.1818)))*43758.5);"
 "}"
 "vec3 rand3(vec3 x)"
 "{"
   "return vec3(rand(x.x*c.xx),rand(x.y*c.xx),rand(x.z*c.xx));"
 "}"
 "mat3 rot(vec3 p)"
 "{"
   "return mat3(c.xyyy,cos(p.x),sin(p.x),0.,-sin(p.x),cos(p.x))*mat3(cos(p.y),0.,-sin(p.y),c.yxy,sin(p.y),0.,cos(p.y))*mat3(cos(p.z),-sin(p.z),0.,sin(p.z),cos(p.z),c.yyyx);"
 "}"
 "vec4 vor(vec3 x)"
 "{"
   "vec3 y=floor(x);"
   "float ret=10.;"
   "vec3 pf=c.yyy,p;"
   "float df=100.,d;"
   "for(int i=-1;i<=1;i+=1)"
     "for(int j=-1;j<=1;j+=1)"
       "for(int k=-1;k<=1;k+=1)"
         "{"
           "p=y+vec3(float(i),float(j),float(k));"
           "p+=rand3(p);"
           "d=length(x-p);"
           "if(d<df)"
             "df=d,pf=p;"
         "}"
   "for(int i=-1;i<=1;i+=1)"
     "for(int j=-1;j<=1;j+=1)"
       "for(int k=-1;k<=1;k+=1)"
         "{"
           "p=y+vec3(float(i),float(j),float(k));"
           "p+=rand3(p);"
           "vec3 o=p-pf;"
           "d=abs(.5-dot(x-pf,o)/length(o));"
           "ret=min(ret,d);"
         "}"
   "return vec4(ret,pf);"
 "}"
 "vec3 vor(vec2 x)"
 "{"
   "vec2 y=floor(x);"
   "float ret=1.;"
   "vec2 pf=c.yy,p;"
   "float df=10.,d;"
   "for(int i=-1;i<=1;i+=1)"
     "for(int j=-1;j<=1;j+=1)"
       "{"
         "p=y+vec2(float(i),float(j));"
         "p+=vec2(rand(p),rand(p+1.));"
         "d=length(x-p);"
         "if(d<df)"
           "df=d,pf=p;"
       "}"
   "for(int i=-1;i<=1;i+=1)"
     "for(int j=-1;j<=1;j+=1)"
       "{"
         "p=y+vec2(float(i),float(j));"
         "p+=vec2(rand(p),rand(p+1.));"
         "vec2 o=p-pf;"
         "d=length(.5*o-dot(x-pf,o)/dot(o,o)*o);"
         "ret=min(ret,d);"
       "}"
   "return vec3(ret,pf);"
 "}"
 "float valuenoise(vec2 x)"
 "{"
   "vec2 y=floor(x);"
   "x=fract(x);"
   "float r00=-1.+2.*rand(y),r10=-1.+2.*rand(y+c.xy),r01=-1.+2.*rand(y+c.yx),r11=-1.+2.*rand(y+c.xx);"
   "return mix(mix(r00,r10,x.x),mix(r01,r11,x.x),x.y);"
 "}"
 "float valuenoise(vec3 x)"
 "{"
   "vec3 y=floor(x);"
   "x=fract(x);"
   "float r000=-1.+2.*rand(y),r100=-1.+2.*rand(y+c.xyy),r010=-1.+2.*rand(y+c.yxy),r001=-1.+2.*rand(y+c.yyx),r110=-1.+2.*rand(y+c.xxy),r011=-1.+2.*rand(y+c.yxx),r101=-1.+2.*rand(y+c.xyx),r111=-1.+2.*rand(y+c.xxx);"
   "return mix(mix(mix(r000,r100,smoothstep(0.,1.,x.x)),mix(r010,r110,smoothstep(0.,1.,x.x)),smoothstep(0.,1.,x.y)),mix(mix(r001,r101,smoothstep(0.,1.,x.x)),mix(r011,r111,smoothstep(0.,1.,x.x)),smoothstep(0.,1.,x.y)),smoothstep(0.,1.,x.z));"
 "}"
 "float mfvaluenoise(vec2 x,float f0,float f1,float phi)"
 "{"
   "float sum=0.,a=1.2;"
   "for(float f=f0;f<f1;f=f*2.)"
     "sum=a*valuenoise(f*x)+sum,a=a*phi;"
   "return sum;"
 "}"
 "vec4 add(vec4 sdf,vec4 sda)"
 "{"
   "return vec4(min(sdf.x,sda.x),mix(sda.yzw,sdf.yzw,smoothstep(-1.5/iResolution.y,1.5/iResolution.y,sda.x)));"
 "}"
 "vec2 add(vec2 sda,vec2 sdb)"
 "{"
   "return mix(sda,sdb,step(sdb.x,sda.x));"
 "}"
 "vec2 sub(vec2 sda,vec2 sdb)"
 "{"
   "return mix(-sda,sdb,step(sda.x,sdb.x));"
 "}"
 "vec4 smoothadd(vec4 sdf,vec4 sda,float a)"
 "{"
   "return vec4(min(sdf.x,sda.x),mix(sda.yzw,sdf.yzw,smoothstep(-a*1.5/iResolution.y,a*1.5/iResolution.y,sda.x)));"
 "}"
 "float lineseg(vec2 x,vec2 p1,vec2 p2)"
 "{"
   "vec2 d=p2-p1;"
   "return length(x-mix(p1,p2,clamp(dot(x-p1,d)/dot(d,d),0.,1.)));"
 "}"
 "float dspiral(vec2 x,float a,float d)"
 "{"
   "float p=atan(x.y,x.x),n=floor((abs(length(x)-a*p)+d*p)/(2.*pi*a));"
   "p+=(n*2.+1.)*pi;"
   "return-abs(length(x)-a*p)+d*p;"
 "}"
 "float circle(vec2 x,float r)"
 "{"
   "return length(x)-r;"
 "}"
 "float circlesegment(vec2 x,float r,float p0,float p1)"
 "{"
   "float p=atan(x.y,x.x);"
   "p=clamp(p,p0,p1);"
   "return length(x-r*vec2(cos(p),sin(p)));"
 "}"
 "float logo(vec2 x,float r)"
 "{"
   "return min(min(circle(x+r*c.zy,r),lineseg(x,r*c.yz,r*c.yx)),circlesegment(x+r*c.xy,r,-.5*pi,.5*pi));"
 "}"
 "float stroke(float d,float w)"
 "{"
   "return abs(d)-w;"
 "}"
 "float dist(vec2 p0,vec2 p1,vec2 p2,vec2 x,float t)"
 "{"
   "return t=clamp(t,0.,1.),length(x-pow(1.-t,2.)*p0-2.*(1.-t)*t*p1-t*t*p2);"
 "}"
 "float length2(in vec2 v)"
 "{"
   "return dot(v,v);"
 "}"
 "float spline2(vec2 p0,vec2 p1,vec2 p2,vec2 x)"
 "{"
   "vec2 bmi=min(p0,min(p1,p2)),bma=max(p0,max(p1,p2)),bce=(bmi+bma)*.5,bra=(bma-bmi)*.5;"
   "float bdi=length2(max(abs(x-bce)-bra,0.));"
   "if(bdi>dmin)"
     "return dmin;"
   "vec2 E=x-p0,F=p2-2.*p1+p0,G=p1-p0;"
   "vec3 ai=vec3(3.*dot(G,F),2.*dot(G,G)-dot(E,F),-dot(E,G))/dot(F,F);"
   "float tau=ai.x/3.,p=ai.y-tau*ai.x,q=-tau*(tau*tau+p)+ai.z,dis=q*q/4.+p*p*p/27.;"
   "if(dis>0.)"
     "{"
       "vec2 ki=-.5*q*c.xx+sqrt(dis)*c.xz,ui=sign(ki)*pow(abs(ki),c.xx/3.);"
       "return dist(p0,p1,p2,x,ui.x+ui.y-tau);"
     "}"
   "float fac=sqrt(-4./3.*p),arg=acos(-.5*q*sqrt(-27./p/p/p))/3.;"
   "vec3 t=c.zxz*fac*cos(arg*c.xxx+c*pi/3.)-tau;"
   "return min(dist(p0,p1,p2,x,t.x),min(dist(p0,p1,p2,x,t.y),dist(p0,p1,p2,x,t.z)));"
 "}"
 "float zextrude(float z,float d2d,float h)"
 "{"
   "vec2 d=abs(vec2(min(d2d,0.),z))-h*c.yx;"
   "return min(max(d.x,d.y),0.)+length(max(d,0.));"
 "}"
 "float rshort(float off)"
 "{"
   "float hilo=mod(off,2.);"
   "off*=.5;"
   "vec2 ind=(vec2(mod(off,iFontWidth),floor(off/iFontWidth))+.05)/iFontWidth;"
   "vec4 block=texture(iFont,ind);"
   "vec2 data=mix(block.xy,block.zw,hilo);"
   "return round(dot(vec2(255.,65280.),data));"
 "}"
 "float dglyph(vec2 x,int ascii)"
 "{"
   "if(ascii==32)"
     "return glyphsize=size*vec2(.02,1.),1.;"
   "float nchars=rshort(0.),off=-1.;"
   "for(float i=0.;i<nchars;i+=1.)"
     "{"
       "int ord=int(rshort(1.+2.*i));"
       "if(ord==ascii)"
         "{"
           "off=rshort(1.+2.*i+1);"
           "break;"
         "}"
     "}"
   "if(off==-1.)"
     "return 1.;"
   "vec2 dx=mix(c.xx,c.zz,vec2(rshort(off),rshort(off+2.)))*vec2(rshort(off+1.),rshort(off+3.));"
   "float npts=rshort(off+4.),xoff=off+5.,yoff=off+6.+npts,toff=off+7.+2.*npts,coff=off+8.+3.*npts,ncont=rshort(coff-1.),d=1.;"
   "vec2 mx=-100.*c.xx,mn=100.*c.xx;"
   "for(float i=0.;i<ncont;i+=1.)"
     "{"
       "float istart=0.,iend=rshort(coff+i);"
       "if(i>0.)"
         "istart=rshort(coff+i-1.)+1.;"
       "vec2 stack[3];"
       "float tstack[3];"
       "int stacksize=0;"
       "for(float j=istart;j<=iend;j+=1.)"
         "{"
           "tstack[stacksize]=rshort(toff+j);"
           "stack[stacksize]=(vec2(rshort(xoff+j),rshort(yoff+j))+dx)/65536.*size;"
           "mx=max(mx,stack[stacksize]);"
           "mn=min(mn,stack[stacksize]);"
           "++stacksize;"
           "if(stacksize==2)"
             "{"
               "if(tstack[0]*tstack[1]==1)"
                 "d=min(d,lineseg(x,stack[0],stack[1])),--j,stacksize=0;"
             "}"
           "else"
             " if(stacksize==3)"
               "{"
                 "if(tstack[0]*tstack[2]==1.)"
                   "d=min(d,spline2(stack[0],stack[1],stack[2],x)),--j,stacksize=0;"
                 "else"
                   "{"
                     "vec2 p=mix(stack[1],stack[2],.5);"
                     "d=min(d,spline2(stack[0],stack[1],p,x));"
                     "stack[0]=p;"
                     "tstack[0]=1.;"
                     "mx=max(mx,stack[0]);"
                     "mn=min(mn,stack[0]);"
                     "--j;"
                     "stacksize=1;"
                   "}"
               "}"
         "}"
       "tstack[stacksize]=rshort(toff+istart);"
       "stack[stacksize]=(vec2(rshort(xoff+istart),rshort(yoff+istart))+dx)/65536.*size;"
       "mx=max(mx,stack[0]);"
       "mn=min(mn,stack[0]);"
       "++stacksize;"
       "if(stacksize==2)"
         "d=min(d,lineseg(x,stack[0],stack[1]));"
       "else"
         " if(stacksize==3)"
           "d=min(d,spline2(stack[0],stack[1],stack[2],x));"
     "}"
   "glyphsize=abs(mx-mn);"
   "return d;"
 "}"
 "float dglyphpts(vec2 x,int ascii)"
 "{"
   "float nchars=rshort(0.),off=-1.;"
   "for(float i=0.;i<nchars;i+=1.)"
     "{"
       "int ord=int(rshort(1.+2.*i));"
       "if(ord==ascii)"
         "{"
           "off=rshort(1.+2.*i+1);"
           "break;"
         "}"
     "}"
   "if(off==-1.)"
     "return 1.;"
   "vec2 dx=mix(c.xx,c.zz,vec2(rshort(off),rshort(off+2.)))*vec2(rshort(off+1.),rshort(off+3.));"
   "float npts=rshort(off+4.),xoff=off+5.,yoff=off+6.+npts,d=1.;"
   "for(float i=0.;i<npts;i+=1.)"
     "{"
       "vec2 xa=(vec2(rshort(xoff+i),rshort(yoff+i))+dx)/65536.*size;"
       "d=min(d,length(x-xa)-.002);"
     "}"
   "return d;"
 "}"
 "mat2 rot(float t)"
 "{"
   "vec2 sc=vec2(cos(t),sin(t));"
   "return mat2(sc*c.xz,sc.yx);"
 "}"
 "float blend(float tstart,float tend,float dt)"
 "{"
   "return smoothstep(tstart-dt,tstart+dt,iTime)*(1.-smoothstep(tend-dt,tend+dt,iTime));"
 "}"
 "vec3 ind;"
 "vec2 scene(vec3 x)"
 "{"
   "x+=iTime*c.yxy*.1-iNBeats*c.xxy;"
   "vec2 dis=12.*vec2((.1+.05*iScale)*valuenoise(x-2.-.2*iTime),(.1+.05*iScale)*valuenoise(x.xy-5.-.2*iTime));"
   "float d=x.z-mfvaluenoise(x.xy-dis,2.,40.,.45+.2*clamp(3.*iScale,0.,1.));"
   "d=max(d,-.5*mfvaluenoise(x.xy-dis,2.,10.,.45+.2*clamp(3.*iScale,0.,1.)));"
   "float dr=.165;"
   "vec3 y=mod(x,dr)-.5*dr;"
   "float guard=-length(max(abs(y)-vec3(.5*dr*c.xx,.6),0.));"
   "guard=abs(guard)+dr*.1;"
   "d=min(d,guard);"
   "return vec2(d,1.);"
 "}"
 "vec2 scene2(vec3 x)"
 "{"
   "vec2 dis=12.*vec2((.1+.05*iScale)*valuenoise(x-2.-.1*iTime),(.1+.05*iScale)*valuenoise(x.xy-5.-.1*iTime));"
   "vec3 v=vor(7.*x.xy-iNBeats-dis);"
   "float h=.4*rand(v.yz+7.+iNBeats)+.2*valuenoise(v.yz+7.+iNBeats)*clamp(5.*iScale,0.,1.)+.1*valuenoise(v.yz-.4*iTime),d=stroke(zextrude(x.z+.1-.2*x.y,stroke(v.x,.15+.15*clamp(3.*iScale,0.,1.)),h),.1),dr=.065;"
   "vec3 y=mod(x,dr)-.5*dr;"
   "float guard=-length(max(abs(y)-vec3(.5*dr*c.xx,.6),0.));"
   "guard=abs(guard)+dr*.1;"
   "d=min(d,guard);"
   "ind=vec3(v.yz,0.);"
   "return add(vec2(d,1.),vec2(x.z,2.));"
 "}"
 "vec2 scene3(vec3 x)"
 "{"
   "x=rot(.05*vec3(1.,2.,3.)*iTime+iNBeats*c.xxx)*x;"
   "vec3 y=mod(x,1.)-.5;"
   "vec4 v=(length(y)-.5)*c.xxxx,w=vor(2.*x-(.2+.1*iScale)*valuenoise(x.xy-2.-iTime));"
   "ind=v.yzw+.1*w.yzw;"
   "float d=max(-stroke(.4*w.x,.005+.001*iScale),stroke(v.x,.1));"
   "d=max(-length(x)+1.,d);"
   "return vec2(abs(d)-.001,1.);"
 "}"
 "vec2 scene4(vec3 x)"
 "{"
   "x=rot(.05*vec3(1.,2.,3.)*iTime+iNBeats)*x;"
   "vec4 v=vor(.5*x-(.1+.05*iScale)*valuenoise(x.xy-2.-iTime)),w=vor(2.*x-(.2+.1*iScale)*valuenoise(x.xy-2.-iTime));"
   "ind=v.yzw+.1*w.yzw;"
   "float d=max(-stroke(.4*w.x,.005+.001*iScale),stroke(v.x,.1));"
   "return vec2(abs(d)-.001,1.);"
 "}\n"
 "#define raymarch(scene,xc,ro,d,dir,s,N,eps,flag)flag=false;for(int i=0;i<N;++i){xc=ro+d*dir;s=scene(xc);if(s.x<eps){flag=true;break;}d+=s.x;}\n"
 "#define calcnormal(scene,n,eps,xc){float ss=scene(xc).x;n=normalize(vec3(scene(xc+eps*c.xyy).xc-ss,scene(xc+eps*c.yxy).xc-ss,scene(xc+eps*c.yyx).xc-ss));}\n"
 "#define camerasetup(camera,ro,r,u,t,uv,dir){camera(ro,r,u,t);t+=uv.x*r+uv.y*u;dir=normalize(t-ro);}\n"
 "#define post(color,uv){col=mix(clamp(col,c.yyy,c.xxx),c.xxx,smoothstep(1.5/iResolution.y,-1.5/iResolution.y,stroke(logo(uv-vec2(-.45,.45),.02),.005)));col+=vec3(0.,0.05,0.1)*sin(uv.y*1050.+5.*iTime);}\n"
 "void camera1(out vec3 ro,out vec3 r,out vec3 u,out vec3 t)"
 "{"
   "ro=c.yyx,r=c.xyy,u=c.yxx,t=c.yxy;"
 "}"
 "vec3 synthcol(float scale,float phase)"
 "{"
   "vec3 c2=vec3(207.,30.,102.)/255.,c3=vec3(245.,194.,87.)/255.;"
   "mat3 r1=rot(.5*phase*vec3(1.1,1.3,1.5));"
   "return 1.1*mix(-cross(c2,r1*c2),-(r1*c2),scale);"
 "}"
 "vec3 stdcolor(vec2 x)"
 "{"
   "return.5+.5*cos(iTime+x.xyx+vec3(0,2,4));"
 "}"
 "vec3 color(float rev,float ln,float index,vec2 uv,vec3 x)"
 "{"
   "vec3 col=c.yyy;"
   "if(index==1.)"
     "{"
       "vec3 c1=stdcolor(x.xy+.5*rand(ind.xy+17.)+iNBeats),c2=stdcolor(x.xy+x.yz+x.zx+.5*rand(ind.xy+12.)+iNBeats+11.+uv),c3=stdcolor(x.xy+x.yz+x.zx+.5*rand(ind.xy+15.)+iNBeats+23.+uv);"
       "col=.1*c1*vec3(1.,1.,1.)+.2*c1*vec3(1.,1.,1.)*ln+vec3(1.,1.,1.)*pow(rev,2.*(2.-1.5*clamp(iScale,0.,1.)))+2.*c1*pow(rev,8.)+3.*c1*pow(rev,16.);"
       "col=clamp(.33*col,0.,1.);"
     "}"
   "else"
     " if(index==2.)"
       "return stdcolor(x.xy+.5*rand(ind.xy+17.)+iNBeats);"
   "return col;"
 "}"
 "vec4 thick(vec2 x,vec4 sdf,vec2 n)"
 "{"
   "for(int i=1;i<6;++i)"
     "sdf=add(vec4(stroke(sdf.x*n.x*n.y*2.*valuenoise((3.+4.*iScale)*x-2.-iTime-1.2),.01),.003/abs(sdf.x+.2*valuenoise(x-2.-iTime))*stdcolor(x+c.xx*.3*float(i))),sdf);"
   "return sdf;"
 "}"
 "vec4 geometry(vec2 x)"
 "{"
   "vec4 sdf=vec4(stroke(stroke(logo(x,.2),.06),.01),2.5*stdcolor(x*1.7));"
   "return sdf;"
 "}"
 "const float dx=.0001;"
 "vec2 normal(vec2 x)"
 "{"
   "float s=geometry(x).x;"
   "return normalize(vec2(geometry(x+dx*c.xy).x-s,geometry(x+dx*c.yx).x-s));"
 "}"
 "void mainImage(out vec4 fragColor,in vec2 fragCoord)"
 "{"
   "vec2 uv=fragCoord/iResolution.yy-.5;"
   "if(iTime<6.)"
     "{"
       "vec4 sdf=vec4(1.,col);"
       "float d=1.,dc=1.,dca=1.;"
       "vec2 vn=.02*vec2(valuenoise(2.3*uv-2.*vec2(1.5,2.4)*iTime),valuenoise(1.7*uv-2.4*vec2(1.2,2.1)*iTime));"
       "{"
         "size=1.54;"
         "carriage=-.25*c.xy;"
         "int str[17]=int[17](72,101,108,108,111,44,32,86,111,114,116,101,120,32,73,73,73);"
         "for(int i=0;i<17;++i)"
           "{"
             "if(abs(uv.x)<1.5&&abs(uv.y)<.1)"
               "{"
                 "vec2 bound=uv-carriage-vn+.05*c.yx;"
                 "d=min(d,dglyph(bound,str[i]));"
                 "float d0=dglyphpts(bound,str[i]);"
                 "dc=min(dc,d0);"
                 "dca=min(dca,stroke(d0,.002));"
                 "carriage+=glyphsize.x*c.xy+.01*c.xy;"
               "}"
           "}"
       "}"
       "d=stroke(d,.0024)+.1*length(vn);"
       "sdf=add(sdf,vec4(d,c.xxx));"
       "sdf=add(sdf,vec4(dca,c.xxx));"
       "sdf=add(sdf,vec4(dc,c.xyy));"
       "col=sdf.yzw*smoothstep(1.5/iResolution.y,-1.5/iResolution.y,sdf.x)*blend(1.,5.,1.);"
     "}"
   "else"
     " if(iTime<11.)"
       "{"
         "vec2 vn=.02*vec2(valuenoise(23.*uv-.5*vec2(1.5,2.4)*iTime),valuenoise(1.7*uv-2.4*vec2(1.2,2.1)*iTime));"
         "float t=iTime-6.,p=7.5*t,k=cos(p),s=sin(p),d=dspiral(mat2(k,s,-s,k)*(uv-vn),.005,.0004);"
         "vec4 sdf=vec4(1.,col);"
         "sdf=add(sdf,vec4(d,c.xyy));"
         "sdf=add(sdf,vec4(stroke(d,.003),c.xxx));"
         "d=stroke(logo(.6*uv-.25*c.xy-vn,.1),.04);"
         "sdf=add(sdf,vec4(d,c.xyy));"
         "sdf=add(sdf,vec4(stroke(d,.004),c.xxx));"
         "col=sdf.yzw*smoothstep(1.5/iResolution.y,-1.5/iResolution.y,sdf.x)*blend(6.,10.,1.);"
       "}"
     "else"
       " if(iTime<15.)"
         "{"
           "vec4 sdf=vec4(1.,col);"
           "float d=1.,dc=1.,dca=1.;"
           "vec2 vn=.02*vec2(valuenoise(2.3*uv-2.*vec2(1.5,2.4)*iTime),valuenoise(1.7*uv-2.4*vec2(1.2,2.1)*iTime));"
           "{"
             "size=1.54;"
             "carriage=-.35*c.xy;"
             "int str[17]=int[17](84,101,97,109,50,49,48,32,99,97,109,101,32,104,101,114,101);"
             "for(int i=0;i<17;++i)"
               "{"
                 "if(abs(uv.x)<1.5&&abs(uv.y)<.1)"
                   "{"
                     "vec2 bound=uv-carriage-vn+.05*c.yx;"
                     "d=min(d,dglyph(bound,str[i]));"
                     "float d0=dglyphpts(bound,str[i]);"
                     "dc=min(dc,d0);"
                     "dca=min(dca,stroke(d0,.002));"
                     "carriage+=glyphsize.x*c.xy+.01*c.xy;"
                   "}"
               "}"
           "}"
           "d=stroke(d,.0024)+.1*length(vn);"
           "sdf=add(sdf,vec4(d,c.xxx));"
           "sdf=add(sdf,vec4(dca,c.xxx));"
           "sdf=add(sdf,vec4(dc,c.xyy));"
           "col=sdf.yzw*smoothstep(1.5/iResolution.y,-1.5/iResolution.y,sdf.x)*blend(12.,14.,1.);"
         "}"
       "else"
         " if(iTime<19.)"
           "{"
             "vec4 sdf=vec4(1.,col);"
             "float d=1.,dc=1.,dca=1.;"
             "vec2 vn=.02*vec2(valuenoise(2.3*uv-2.*vec2(1.5,2.4)*iTime),valuenoise(1.7*uv-2.4*vec2(1.2,2.1)*iTime));"
             "{"
               "size=1.54;"
               "carriage=-.3*c.xy;"
               "int str[17]=int[17](116,111,32,112,97,114,116,121,32,119,105,116,104,32,121,111,117);"
               "for(int i=0;i<17;++i)"
                 "{"
                   "if(abs(uv.x)<1.5&&abs(uv.y)<.1)"
                     "{"
                       "vec2 bound=uv-carriage-vn+.05*c.yx;"
                       "d=min(d,dglyph(bound,str[i]));"
                       "float d0=dglyphpts(bound,str[i]);"
                       "dc=min(dc,d0);"
                       "dca=min(dca,stroke(d0,.002));"
                       "carriage+=glyphsize.x*c.xy+.01*c.xy;"
                     "}"
                 "}"
             "}"
             "d=stroke(d,.0024)+.1*length(vn);"
             "sdf=add(sdf,vec4(d,c.xxx));"
             "sdf=add(sdf,vec4(dca,c.xxx));"
             "sdf=add(sdf,vec4(dc,c.xyy));"
             "col=sdf.yzw*smoothstep(1.5/iResolution.y,-1.5/iResolution.y,sdf.x)*blend(16.,18.,1.);"
           "}"
         "else"
           " if(iTime<50.)"
             "{"
               "vec3 ro,r,u,t,x,dir;"
               "camerasetup(camera1,ro,r,u,t,uv,dir);"
               "float d=-(ro.z-1.)/dir.z;"
               "bool hit;"
               "vec2 s;"
               "raymarch(scene,x,ro,d,dir,s,300,.0001,hit);"
               "if(hit==false)"
                 "{"
                   "post(col,uv);"
                   "fragColor=vec4(col,1.);"
                   "return;"
                 "}"
               "vec3 n;"
               "calcnormal(scene,n,.001,x);"
               "vec3 l=x+2.*c.yyx,re=normalize(reflect(-l,n)),v=normalize(x-ro);"
               "float rev=abs(dot(re,v)),ln=abs(dot(l,n));"
               "col=color(rev,ln,s.y,uv,x);"
               "for(float i=.7;i>=.5;i-=.2)"
                 "{"
                   "dir=normalize(reflect(dir,n));"
                   "d=.5;"
                   "ro=x;"
                   "raymarch(scene,x,ro,d,dir,s,300,.005,hit);"
                   "if(hit==false)"
                     "{"
                       "post(col,uv);"
                       "fragColor=vec4(col,1.);"
                       "break;"
                     "}"
                   "calcnormal(scene,n,.001,x);"
                   "l=x+2.*c.yyx;"
                   "re=normalize(reflect(-l,n));"
                   "v=normalize(x-ro);"
                   "rev=abs(dot(re,v));"
                   "ln=abs(dot(l,n));"
                   "col=mix(col,color(rev,ln,s.y,uv,x),i);"
                 "}"
             "}"
           "else"
             " if(iTime<54.)"
               "{"
                 "vec4 sdf=vec4(1.,col);"
                 "float d=1.,dc=1.,dca=1.;"
                 "vec2 vn=.02*vec2(valuenoise(2.3*uv-2.*vec2(1.5,2.4)*iTime),valuenoise(1.7*uv-2.4*vec2(1.2,2.1)*iTime));"
                 "{"
                   "size=1.54;"
                   "carriage=-.3*c.xy;"
                   "int str[16]=int[16](67,111,100,101,32,58,58,32,78,82,52,32,38,32,81,77);"
                   "for(int i=0;i<16;++i)"
                     "{"
                       "if(abs(uv.x)<1.5&&abs(uv.y)<.1)"
                         "{"
                           "vec2 bound=uv-carriage-vn+.05*c.yx;"
                           "d=min(d,dglyph(bound,str[i]));"
                           "float d0=dglyphpts(bound,str[i]);"
                           "dc=min(dc,d0);"
                           "dca=min(dca,stroke(d0,.002));"
                           "carriage+=glyphsize.x*c.xy+.01*c.xy;"
                         "}"
                     "}"
                 "}"
                 "d=stroke(d,.0024)+.1*length(vn);"
                 "sdf=add(sdf,vec4(d,c.xxx));"
                 "sdf=add(sdf,vec4(dca,c.xxx));"
                 "sdf=add(sdf,vec4(dc,c.xyy));"
                 "col=sdf.yzw*smoothstep(1.5/iResolution.y,-1.5/iResolution.y,sdf.x)*blend(51.,53.,1.);"
               "}"
             "else"
               " if(iTime<58.)"
                 "{"
                   "vec4 sdf=vec4(1.,col);"
                   "float d=1.,dc=1.,dca=1.;"
                   "vec2 vn=.02*vec2(valuenoise(2.3*uv-2.*vec2(1.5,2.4)*iTime),valuenoise(1.7*uv-2.4*vec2(1.2,2.1)*iTime));"
                   "{"
                     "size=1.54;"
                     "carriage=-.3*c.xy;"
                     "int str[16]=int[16](71,70,88,32,58,58,32,78,82,52,32,38,32,65,87,69);"
                     "for(int i=0;i<16;++i)"
                       "{"
                         "if(abs(uv.x)<1.5&&abs(uv.y)<.1)"
                           "{"
                             "vec2 bound=uv-carriage-vn+.05*c.yx;"
                             "d=min(d,dglyph(bound,str[i]));"
                             "float d0=dglyphpts(bound,str[i]);"
                             "dc=min(dc,d0);"
                             "dca=min(dca,stroke(d0,.002));"
                             "carriage+=glyphsize.x*c.xy+.01*c.xy;"
                           "}"
                       "}"
                   "}"
                   "d=stroke(d,.0024)+.1*length(vn);"
                   "sdf=add(sdf,vec4(d,c.xxx));"
                   "sdf=add(sdf,vec4(dca,c.xxx));"
                   "sdf=add(sdf,vec4(dc,c.xyy));"
                   "col=sdf.yzw*smoothstep(1.5/iResolution.y,-1.5/iResolution.y,sdf.x)*blend(55.,57.,1.);"
                 "}"
               "else"
                 " if(iTime<62.)"
                   "{"
                     "vec4 sdf=vec4(1.,col);"
                     "float d=1.,dc=1.,dca=1.;"
                     "vec2 vn=.02*vec2(valuenoise(2.3*uv-2.*vec2(1.5,2.4)*iTime),valuenoise(1.7*uv-2.4*vec2(1.2,2.1)*iTime));"
                     "{"
                       "size=1.54;"
                       "carriage=-.1*c.xy;"
                       "int str[9]=int[9](83,70,88,32,58,58,32,81,77);"
                       "for(int i=0;i<9;++i)"
                         "{"
                           "if(abs(uv.x)<1.5&&abs(uv.y)<.1)"
                             "{"
                               "vec2 bound=uv-carriage-vn+.05*c.yx;"
                               "d=min(d,dglyph(bound,str[i]));"
                               "float d0=dglyphpts(bound,str[i]);"
                               "dc=min(dc,d0);"
                               "dca=min(dca,stroke(d0,.002));"
                               "carriage+=glyphsize.x*c.xy+.01*c.xy;"
                             "}"
                         "}"
                     "}"
                     "d=stroke(d,.0024)+.1*length(vn);"
                     "sdf=add(sdf,vec4(d,c.xxx));"
                     "sdf=add(sdf,vec4(dca,c.xxx));"
                     "sdf=add(sdf,vec4(dc,c.xyy));"
                     "col=sdf.yzw*smoothstep(1.5/iResolution.y,-1.5/iResolution.y,sdf.x)*blend(59.,61.,1.);"
                   "}"
                 "else"
                   " if(iTime<66.)"
                     "{"
                       "vec4 sdf=vec4(1.,col);"
                       "float d=1.,dc=1.,dca=1.;"
                       "vec2 vn=.02*vec2(valuenoise(2.3*uv-2.*vec2(1.5,2.4)*iTime),valuenoise(1.7*uv-2.4*vec2(1.2,2.1)*iTime));"
                       "{"
                         "size=1.54;"
                         "carriage=-.2*c.xy;"
                         "int str[14]=int[14](70,101,97,116,46,32,76,101,32,77,105,113,117,101);"
                         "for(int i=0;i<14;++i)"
                           "{"
                             "if(abs(uv.x)<1.5&&abs(uv.y)<.1)"
                               "{"
                                 "vec2 bound=uv-carriage-vn+.05*c.yx;"
                                 "d=min(d,dglyph(bound,str[i]));"
                                 "float d0=dglyphpts(bound,str[i]);"
                                 "dc=min(dc,d0);"
                                 "dca=min(dca,stroke(d0,.002));"
                                 "carriage+=glyphsize.x*c.xy+.01*c.xy;"
                               "}"
                           "}"
                       "}"
                       "d=stroke(d,.0024)+.1*length(vn);"
                       "sdf=add(sdf,vec4(d,c.xxx));"
                       "sdf=add(sdf,vec4(dca,c.xxx));"
                       "sdf=add(sdf,vec4(dc,c.xyy));"
                       "col=sdf.yzw*smoothstep(1.5/iResolution.y,-1.5/iResolution.y,sdf.x)*blend(63.,65.,1.);"
                     "}"
                   "else"
                     " if(iTime<76.)"
                       "{"
                         "vec3 ro,r,u,t,x,dir;"
                         "camerasetup(camera1,ro,r,u,t,uv,dir);"
                         "float d=-(ro.z-.5-.3*clamp(5.*iScale,0.,1.))/dir.z;"
                         "bool hit;"
                         "vec2 s;"
                         "raymarch(scene2,x,ro,d,dir,s,250,.0001,hit);"
                         "if(hit==false)"
                           "{"
                             "post(col,uv);"
                             "fragColor=vec4(col,1.);"
                             "return;"
                           "}"
                         "vec3 n;"
                         "calcnormal(scene2,n,.001,x);"
                         "vec3 l=x+2.*c.yyx,re=normalize(reflect(-l,n)),v=normalize(x-ro);"
                         "float rev=abs(dot(re,v)),ln=abs(dot(l,n));"
                         "col=color(rev,ln,s.y,uv,x);"
                         "for(float i=.7;i>=.3;i-=.2)"
                           "{"
                             "dir=normalize(reflect(dir,n));"
                             "d=.005;"
                             "ro=x;"
                             "raymarch(scene2,x,ro,d,dir,s,35,.0005,hit);"
                             "if(hit==false)"
                               "{"
                                 "post(col,uv);"
                                 "fragColor=vec4(col,1.);"
                                 "break;"
                               "}"
                             "calcnormal(scene2,n,.001,x);"
                             "l=x+2.*c.yyx;"
                             "re=normalize(reflect(-l,n));"
                             "v=normalize(x-ro);"
                             "rev=abs(dot(re,v));"
                             "ln=abs(dot(l,n));"
                             "col=mix(col,color(rev,ln,s.y,uv,x),i);"
                           "}"
                         "col=mix(col,c.yyy,tanh(.2*abs(x.y+x.z)));"
                       "}"
                     "else"
                       " if(iTime<86.)"
                         "{"
                           "vec3 ro=c.yyx,r=c.xyy,u=c.yxy,t=c.yxy+uv.x*r+uv.y*u,x,dir=normalize(t-ro);"
                           "float d=1.;"
                           "bool hit;"
                           "vec2 s;"
                           "raymarch(scene3,x,ro,d,dir,s,80,.0001,hit);"
                           "if(hit==false)"
                             "{"
                               "post(col,uv);"
                               "fragColor=vec4(col,1.);"
                               "return;"
                             "}"
                           "vec3 n;"
                           "calcnormal(scene3,n,.005,x);"
                           "vec3 l=x+c.xxx,re=normalize(reflect(-l,n)),v=normalize(x-ro),c1=stdcolor(uv+.5*ind.x+iNBeats),c2=stdcolor(uv+.5*ind.y+iNBeats),c3=stdcolor(uv+.5*ind.z+iNBeats);"
                           "float rev=abs(dot(re,v)),ln=abs(dot(l,n));"
                           "if(s.y==1.)"
                             "col=.1*c1*vec3(1.,.3,.3)+.2*c1*vec3(1.,.3,.3)*ln+vec3(1.,1.,.1)*pow(rev,2.*(2.-1.5*clamp(iScale,0.,1.)))+2.*c1*pow(rev,8.)+3.*c1*pow(rev,16.),col=abs(col);"
                           "col=clamp(.33*col,0.,1.);"
                         "}"
                       "else"
                         " if(iTime<96.)"
                           "{"
                             "vec3 ro=c.yyx,r=c.xyy,u=c.yxy,t=c.yxy+uv.x*r+uv.y*u,x,dir=normalize(t-ro);"
                             "float d=1.;"
                             "bool hit;"
                             "vec2 s;"
                             "raymarch(scene4,x,ro,d,dir,s,80,.0001,hit);"
                             "if(hit==false)"
                               "{"
                                 "post(col,uv);"
                                 "fragColor=vec4(col,1.);"
                                 "return;"
                               "}"
                             "vec3 n;"
                             "calcnormal(scene4,n,.005,x);"
                             "vec3 l=x+c.xxx,re=normalize(reflect(-l,n)),v=normalize(x-ro),c1=stdcolor(uv+.5*ind.x+iNBeats),c2=stdcolor(uv+.5*ind.y+iNBeats),c3=stdcolor(uv+.5*ind.z+iNBeats);"
                             "float rev=abs(dot(re,v)),ln=abs(dot(l,n));"
                             "if(s.y==1.)"
                               "col=.1*c1*vec3(1.,.3,.3)+.2*c1*vec3(1.,.3,.3)*ln+vec3(1.,1.,.1)*pow(rev,2.*(2.-1.5*clamp(iScale,0.,1.)))+2.*c1*pow(rev,8.)+3.*c1*pow(rev,16.),col=abs(col);"
                             "col=clamp(.33*col,0.,1.);"
                           "}"
                         "else"
                           " if(iTime<100.)"
                             "{"
                               "vec4 sdf=vec4(1.,col);"
                               "float d=1.,dc=1.,dca=1.;"
                               "vec2 vn=.02*vec2(valuenoise(2.3*uv-2.*vec2(1.5,2.4)*iTime),valuenoise(1.7*uv-2.4*vec2(1.2,2.1)*iTime));"
                               "{"
                                 "size=1.54;"
                                 "carriage=-.45*c.xy;"
                                 "int str[21]=int[21](83,97,110,116,97,32,104,97,116,115,32,97,114,101,32,116,105,112,112,101,100);"
                                 "for(int i=0;i<21;++i)"
                                   "{"
                                     "if(abs(uv.x)<1.5&&abs(uv.y)<.1)"
                                       "{"
                                         "vec2 bound=uv-carriage-vn+.05*c.yx;"
                                         "d=min(d,dglyph(bound,str[i]));"
                                         "float d0=dglyphpts(bound,str[i]);"
                                         "dc=min(dc,d0);"
                                         "dca=min(dca,stroke(d0,.002));"
                                         "carriage+=glyphsize.x*c.xy+.01*c.xy;"
                                       "}"
                                   "}"
                               "}"
                               "d=stroke(d,.0024)+.1*length(vn);"
                               "sdf=add(sdf,vec4(d,c.xxx));"
                               "sdf=add(sdf,vec4(dca,c.xxx));"
                               "sdf=add(sdf,vec4(dc,c.xyy));"
                               "col=sdf.yzw*smoothstep(1.5/iResolution.y,-1.5/iResolution.y,sdf.x)*blend(97.,99.,1.);"
                             "}"
                           "else"
                             " if(iTime<104.)"
                               "{"
                                 "vec4 sdf=vec4(1.,col);"
                                 "float d=1.,dc=1.,dca=1.;"
                                 "vec2 vn=.02*vec2(valuenoise(2.3*uv-2.*vec2(1.5,2.4)*iTime),valuenoise(1.7*uv-2.4*vec2(1.2,2.1)*iTime));"
                                 "{"
                                   "size=1.54;"
                                   "carriage=-.35*c.xy;"
                                   "int str[17]=int[17](70,111,114,32,75,101,119,108,101,114,115,32,38,32,77,70,88);"
                                   "for(int i=0;i<17;++i)"
                                     "{"
                                       "if(abs(uv.x)<1.5&&abs(uv.y)<.1)"
                                         "{"
                                           "vec2 bound=uv-carriage-vn+.05*c.yx;"
                                           "d=min(d,dglyph(bound,str[i]));"
                                           "float d0=dglyphpts(bound,str[i]);"
                                           "dc=min(dc,d0);"
                                           "dca=min(dca,stroke(d0,.002));"
                                           "carriage+=glyphsize.x*c.xy+.01*c.xy;"
                                         "}"
                                     "}"
                                 "}"
                                 "d=stroke(d,.0024)+.1*length(vn);"
                                 "sdf=add(sdf,vec4(d,c.xxx));"
                                 "sdf=add(sdf,vec4(dca,c.xxx));"
                                 "sdf=add(sdf,vec4(dc,c.xyy));"
                                 "col=sdf.yzw*smoothstep(1.5/iResolution.y,-1.5/iResolution.y,sdf.x)*blend(101.,103.,1.);"
                               "}"
                             "else"
                               " if(iTime<108.)"
                                 "{"
                                   "vec4 sdf=vec4(1.,col);"
                                   "float d=1.,dc=1.,dca=1.;"
                                   "vec2 vn=.02*vec2(valuenoise(2.3*uv-2.*vec2(1.5,2.4)*iTime),valuenoise(1.7*uv-2.4*vec2(1.2,2.1)*iTime));"
                                   "{"
                                     "size=1.54;"
                                     "carriage=-.35*c.xy;"
                                     "int str[16]=int[16](74,117,109,97,108,97,117,116,97,32,38,32,114,103,98,97);"
                                     "for(int i=0;i<16;++i)"
                                       "{"
                                         "if(abs(uv.x)<1.5&&abs(uv.y)<.1)"
                                           "{"
                                             "vec2 bound=uv-carriage-vn+.05*c.yx;"
                                             "d=min(d,dglyph(bound,str[i]));"
                                             "float d0=dglyphpts(bound,str[i]);"
                                             "dc=min(dc,d0);"
                                             "dca=min(dca,stroke(d0,.002));"
                                             "carriage+=glyphsize.x*c.xy+.01*c.xy;"
                                           "}"
                                       "}"
                                   "}"
                                   "d=stroke(d,.0024)+.1*length(vn);"
                                   "sdf=add(sdf,vec4(d,c.xxx));"
                                   "sdf=add(sdf,vec4(dca,c.xxx));"
                                   "sdf=add(sdf,vec4(dc,c.xyy));"
                                   "col=sdf.yzw*smoothstep(1.5/iResolution.y,-1.5/iResolution.y,sdf.x)*blend(105.,107.,1.);"
                                 "}"
                               "else"
                                 " if(iTime<128.)"
                                   "{"
                                     "vec2 x=uv+.1*vec2(valuenoise(uv-5.*iTime),valuenoise(uv-2.-5.*iTime));"
                                     "vec4 sdf=geometry(x);"
                                     "vec2 n=normal(x);"
                                     "sdf=thick(x,sdf,n);"
                                     "col=sdf.yzw*smoothstep(1.5/iResolution.y,-1.5/iResolution.y,sdf.x)*blend(111.,126.,1.);"
                                   "}"
   "post(col,uv);"
   "fragColor=vec4(col,1.);"
 "}"
 "void main()"
 "{"
   "mainImage(gl_FragColor,gl_FragCoord.xy);"
 "}";

#endif // GFX_H_
