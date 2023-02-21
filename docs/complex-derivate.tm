<TeXmacs|2.1>

<style|generic>

<\body>
  <\strong>
    How to calculate derivates of complex valued functions and their partial
    component-wise derivates
  </strong>

  <em|Tomas Ukkonen, Novel Insight Research, 2023>

  Calculating partial derivates <math|<frac|\<partial\>x|\<partial\>\<b-z\>>>
  and <math|<frac|\<partial\>y|\<partial\>\<b-z\>>> of complex valued
  function <math|f<around*|(|\<b-z\>|)>=x<around*|(|\<b-z\>|)>+i*y<around*|(|\<b-z\>|)>>
  is non-trivial calculation because scaling factor is not 1 but
  <math|<frac|1|2>>. I show two ways to calculate these derivates, first one
  uses Wirtinger calculus and other one Cauchy-Riemann equations. These
  results are well-cited in the literature but this paper works are
  reference/memore referesher.

  <with|font-series|bold|Wirtinger calculus>

  You can solve for partial derivate of <math|<frac|\<partial\>|\<partial\>\<b-z\>>>
  by solving the following linear equations. We derivate function using its
  component functions:

  <math|<frac|\<partial\>|\<partial\>x>=<frac|\<partial\>\<b-z\>|\<partial\>x>*<frac|\<partial\>|\<partial\>\<b-z\>>+<frac|\<partial\><wide|\<b-z\>|\<bar\>>|\<partial\>x>*<frac|\<partial\>|\<partial\><wide|\<b-z\>|\<bar\>>>=*<frac|\<partial\>|\<partial\>\<b-z\>>+<frac|\<partial\>|\<partial\><wide|\<b-z\>|\<bar\>>>>

  <math|<frac|\<partial\>|\<partial\>y>=<frac|\<partial\>\<b-z\>|\<partial\>y>*<frac|\<partial\>|\<partial\>\<b-z\>>+<frac|\<partial\><wide|\<b-z\>|\<bar\>>|\<partial\>y>*<frac|\<partial\>|\<partial\><wide|\<b-z\>|\<bar\>>>=*i*<frac|\<partial\>|\<partial\>\<b-z\>>-i*<frac|\<partial\>|\<partial\><wide|\<b-z\>|\<bar\>>>>

  This gives linear equation which inverse we can solve and solve for
  <math|<frac|\<partial\>|\<partial\>\<b-z\>>> (and
  <math|<frac|\<partial\>|\<partial\><wide|\<b-z\>|\<bar\>>>>),

  <math|<matrix|<tformat|<table|<row|<cell|<frac|\<partial\>|\<partial\>x>>>|<row|<cell|<frac|\<partial\>|\<partial\>y>>>>>>=<matrix|<tformat|<table|<row|<cell|1>|<cell|1>>|<row|<cell|i>|<cell|-i>>>>><matrix|<tformat|<table|<row|<cell|<frac|\<partial\>|\<partial\>\<b-z\>>>>|<row|<cell|<frac|\<partial\>|\<partial\><wide|\<b-z\>|\<bar\>>>>>>>>>
  and matrix inverse is

  <math|<matrix|<tformat|<table|<row|<cell|<frac|\<partial\>|\<partial\>\<b-z\>>>>|<row|<cell|<frac|\<partial\>|\<partial\><wide|\<b-z\>|\<bar\>>>>>>>>=<frac|1|2><matrix|<tformat|<table|<row|<cell|1>|<cell|-i>>|<row|<cell|1>|<cell|+i>>>>>><math|<matrix|<tformat|<table|<row|<cell|<frac|\<partial\>|\<partial\>x>>>|<row|<cell|<frac|\<partial\>|\<partial\>y>>>>>>>,
  which gives familiar Wirtinger formula <math|<frac|\<partial\>|\<partial\>\<b-z\>>=<frac|1|2><around*|(|<frac|\<partial\>|\<partial\>x>-i*<frac|\<partial\>|\<partial\>y>|)>>.

  \;

  <with|font-series|bold|Cauchy-Riemann equations>

  Cauchy-Riemann equations are solved by assuming that derivates to complex
  numbered functions must be same if point of derivation is approached from
  any direction from the complex-plane. This means derivate must be also same
  if if approach the point from x-axis (real-value) and y-axis
  (imaginary-value).

  Cauchy-Riemann equations <math|f<around*|(|x,y|)>=u<around*|(|x,y|)>+i*v<around*|(|x,y|)>*>
  are:

  <math|u<rsub|x>=v<rsub|y>,v<rsub|x>=-u<rsub|y>>

  If we plug these to our Wirtinger formula we get

  <math|<frac|\<partial\>*f|\<partial\>\<b-z\>>=<frac|1|2><around*|(|<frac|\<partial\>*f|\<partial\>x>-i*<frac|\<partial\>*f|\<partial\>y>|)>=<frac|1|2><around*|(|u<rsub|x>+i*v<rsub|x>-i*u<rsub|y>+v<rsub|y>|)>=<frac|1|2><around*|(|u<rsub|x>+i*v<rsub|x>+i*v<rsub|x>+u<rsub|x>|)>=u<rsub|x>+i*v<rsub|x>=f<rprime|'><around*|(|\<b-z\>|)>>

  \;

  This is so because we get derivate along x-axis which must be correct if
  <math|f<around*|(|\<b-z\>|)>> is differentiable so derivation from any
  direction gives the correct result.

  \;

  However, there these calculations are a bit formal and there is no
  intuitive solution which shows why we must scale derivates using
  <math|<frac|1|2>> and not just use straight-forward partial derivation
  calculation which gives wrong equation <math|<frac|\<partial\>|\<partial\>\<b-z\>>=<around*|(|<frac|\<partial\>|\<partial\>x>-i*<frac|\<partial\>|\<partial\>y>|)>>,
  which gives wrong scaling to the derivate.

  \;

  The correct formula is <math|<frac|\<partial\>*f|\<partial\>\<b-z\>>=<frac|1|2><around*|(|<frac|\<partial\>*f|\<partial\>x>-i*<frac|\<partial\>*f|\<partial\>y>|)>>.
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>