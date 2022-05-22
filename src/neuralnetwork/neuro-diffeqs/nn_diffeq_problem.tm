<TeXmacs|2.1>

<style|generic>

<\body>
  <strong|Attempt to solve non-linear differential equation models to predict
  system responses>

  <\padded-center>
    <\em>
      Tomas Ukkonen, Novel Insight, 2022

      tomas.ukkonen@iki.fi
    </em>
  </padded-center>

  Predicting brain EEG responses might benefit from differential equation
  model which predicts what happens next in the brain normally. The
  differential equation model uses neural nets as non-linear function.

  <\padded-center>
    <math|<frac|d\<b-x\>|d*t>=\<b-f\><around*|(|\<b-x\><around*|(|t|)>,\<b-z\>|)>>,
    where <math|\<b-z\>> is outside stimulus,
    <math|<frac|d\<b-z\>|d*t>\<approx\>small>
  </padded-center>

  This may be a bit difficult problem so we initially simplify the problem.

  <\padded-center>
    <math|<frac|d\<b-x\>|d*t>=\<b-f\><around*|(|\<b-x\><around*|(|t|)>|)>>,
    where <math|\<b-f\><around*|(|\<b-x\>|)>> is neural net
  </padded-center>

  This differential equation model can be easily numerically simulated using,
  for example, Runge-Kutta integration method
  <math|\<b-x\><around*|(|t|)>=\<b-x\><rsub|0>+<big|int>\<b-f\><around*|(|\<b-x\><around*|(|t|)>|)>d*t>.

  <strong|MSE minimization>

  For fitting neural network parameters <math|\<b-w\>> to data
  <math|<around*|{|<around*|(|\<b-y\><rsub|i>,t<rsub|i>|)>|}>>, we can
  minimize for least squares:

  <\padded-center>
    <math|e<around*|(|\<b-w\>|)>=<frac|1|2*N><big|sum><rsup|N><rsub|i=1><around*|(|\<b-x\><around*|(|t<rsub|i>,\<b-w\>|)>-\<b-y\><rsub|i>|)><rsup|<rsup|T>><around*|(|\<b-x\><around*|(|t<rsub|i>,\<b-w\>|)>-\<b-y\><rsub|i>|)><rsup|>>

    <math|\<nabla\><rsub|\<b-w\>>*e<around*|(|\<b-w\>|)>=<frac|1|N><big|sum><rsup|N><rsub|i=1><around*|(|\<b-x\><around*|(|t<rsub|i>,\<b-w\>|)>-\<b-y\><rsub|i>|)><rsup|<rsup|T>><big|int><rsub|0><rsup|t<rsub|i>><rsup|<rsup|>>\<nabla\><rsub|\<b-w\>>\<b-f\><around*|(|\<b-x\><around*|(|t|)>|)>d*t>
  </padded-center>

  This is difficult to solve(?) but we can use Runge-Kutta to solve for
  <math|\<b-x\><around*|(|t<rsub|i>|)>> terms. After this we can do
  Runge-Kutta for each weight <math|\<b-w\><rsub|k>> in gradient term
  <math|\<nabla\><rsub|\<b-w\>>\<b-f\><around*|(|\<b-x\><around*|(|0|)>|)>>
  and substract <math|\<b-x\><rsub|0>> term from the results to calculate the
  gradient matrix term(?).

  <strong|Bayesian Hamiltonian Monte Carlo sampler>

  Because it is difficult to use Runge-Kutta repeatedly. It might be easier
  to just use Runge-Kutta once to estimate values
  <math|\<b-x\><around*|(|t<rsub|i>|)>>. We can then define gaussian error
  function with zero mean and unit variance error term. This means we have a
  model <math|p<around*|(|data<around*|\|||\|>\<b-w\>|)>\<approx\>Exp<around*|(|-error<around*|(|\<b-w\>|)>|)>>
  which we can reverse by using bayesian rule and assume flat constant prior
  for weight values. This is then maximum likelihood of data method.

  Once we have defined probability distribution we can select random starting
  point <math|\<b-w\><rsub|0>> near origo and sample using Hamiltonian Monte
  Carlo sampler and select maximum probability weight <math|\<b-w\>> that is
  found.

  The exact probability function is:\ 

  <\padded-center>
    <math|p<around*|(|\<b-w\><around*|\|||\|>data|)>\<propto\>p<around*|(|data<around*|\|||\|>\<b-w\>|)>\<propto\>Exp<around*|(|-<frac|1|2*N><big|sum><rsup|N><rsub|i=1><around*|(|\<b-x\><around*|(|t<rsub|i>,\<b-w\>|)>-\<b-y\><rsub|i>|)><rsup|<rsup|T>><around*|(|\<b-x\><around*|(|t<rsub|i>,\<b-w\>|)>-\<b-y\><rsub|i>|)>|)>>
  </padded-center>

  And we use Runge-Kutta to simulate <math|\<b-x\><around*|(|t<rsub|i>|)>>
  values every time <math|\<b-w\>> changes. In practice, we need also match
  time values <math|t<rsub|i>> to Runge-Kutta step intervals and we either
  linearly interpolate <math|\<b-x\>> values from too closest time steps or
  select <math|\<b-x\>> which time value is the closest (initially try this).

  <with|font-shape|italic|NEWS: HMC sampler cannot find good results but
  11-layer deep neural net's error becomes slowly smaller (several hours) so
  it could maybe find something.>\ 

  <strong|Expectation Maximization>

  Expectation maximization techniques are recommended for the problem. Find
  more information.

  <with|font-series|bold|PLAN>

  More study.

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>