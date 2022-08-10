<TeXmacs|2.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Stock Market Short-Term
  Prediction>|<doc-author|<author-data|<author-name|Tomas
  Ukkonen>|<\author-affiliation>
    Novel Insight Research
  </author-affiliation>|<\author-affiliation>
    tomas.ukkonen@novelinsight.fi
  </author-affiliation>>>>

  <abstract-data|<abstract|This is second attemp to do stock market
  prediction. The first attempt by trying to use historical price data and
  deep 20-layer residual neural net failed because minimal neural network
  (1-layer) gives similar thought a bit less reliable results. Bayesian
  Neural Networks used together with Hamiltonian Monte Carlo sampling does
  not give very accurate results when predicting uncertainty in the next day
  stock market data. Further attempt to do 10 or more probabilistic
  computations of neural net's weights and using Markov consensus vote
  counting to combine results seem to be a bit useful. To get better results,
  more complex method will be attempted.>>

  <section|Bayesian Neural Network Differential Equation Model>

  Bayesian methods are used to learn differential equation model from stock
  market data. Probabilistic methods to handle uncertainty is used because
  differential equations are sensitive to noise in signals. We attempt to
  learn non-linear neural net based differential equation:
  <math|<frac|\<partial\>\<b-x\><around*|(|t|)>|\<partial\>t>=f<around*|(|\<b-x\><around*|(|t|)>,\<b-w\>|)>>.
  The future values of <math|\<b-x\><around*|(|t|)>> are simulated using
  initial conditions <math|\<b-x\><around*|(|0|)>> and Runge-Kutta simulation
  method. These values are compared to measurements
  <math|\<b-x\><rsub|h><around*|(|t|)>> (historical data from the stock
  market). It is difficult or impossible to calculate gradient of the neural
  net, therefore sampling methods are attemped. We assume normally
  distributed error and minimize squared error. For efficient Hamiltonian
  Monte Carlo (HMC) sampling, we need also some kind of gradient of the
  squared error term.

  <\padded-center>
    <math|E<around*|(|\<b-w\>|)>=<frac|1|2>*<big|sum><rsub|i><around*|\<\|\|\>|\<b-x\><around*|(|t<rsub|i>|)>-\<b-x\><rsub|h><around*|(|t<rsub|i>|)>|\<\|\|\>><rsup|2>+\<varepsilon\>>,\ 

    <math|\<nabla\><rsub|\<b-w\>>E<around*|(|\<b-w\>|)>=<big|sum><rsub|i><around*|(|<big|int><rsup|t<rsub|i>><rsub|0><around*|(|\<b-x\><around*|(|t<rsub|i>|)>-\<b-x\><rsub|h><around*|(|t<rsub|i>|)>|)><rsup|T>\<nabla\><rsub|\<b-w\>>f<around*|(|\<b-x\><around*|(|t|)>,\<b-w\>|)>d*t|)>>
  </padded-center>

  Here we can use Runge-Kutta again to estimate gradient term:
  <math|<frac|\<partial\><around*|(|\<nabla\><rsub|\<b-w\>>\<b-x\><around*|(|t|)>|)>|\<partial\>t>>.
  Also the weight vector is assumed to be normally distributed so we get a
  posterior <math|p<around*|(|\<b-w\><around*|\|||\|><with|font-series|bold|data>|)>>
  from which we can sample.

  <subsection|Runge-Kutta for Jacobian/Gradient>

  In practice, we need to implement Runge-Kutta to estimate integrated
  gradient terms <math|<frac|\<partial\><around*|(|<around*|(|\<b-x\><around*|(|t<rsub|i>|)>-\<b-x\><rsub|h><around*|(|t<rsub|i>|)>|)><rsup|T>\<nabla\><rsub|\<b-w\>>\<b-x\><around*|(|t<rsub|i>|)>|)>|\<partial\>t>=<frac|\<partial\>\<b-x\><rprime|'><rsub|i>|\<partial\>*t>>
  and we know values of <math|\<b-x\><rsub|i>> from the basic Runge-Kutta
  simulation. Runge-Kutta simulation formulas are:

  <math|\<b-y\><rsub|n+1>=\<b-y\><rsub|n>+<frac|1|6><around*|(|\<b-k\><rsub|1>+2*\<b-k\><rsub|2>+2*\<b-k\><rsub|3>+\<b-k\><rsub|4>|)>*h>,
  <math|\<b-y\><rsub|n>=\<b-x\><rprime|'><around*|(|t<rsub|n>|)>>,
  <math|\<b-y\><rsub|0>=\<b-x\><rsub|0>>

  <math|t<rsub|n+1>=t<rsub|n>+h>

  <math|\<b-k\><rsub|1>=\<b-g\><around*|(|t<rsub|n>,\<b-y\><rsub|n>|)>>,
  <math|\<b-k\><rsub|2>=\<b-g\><around*|(|t<rsub|n>+<frac|h|2>,\<b-y\><rsub|n>+h*<frac|\<b-k\><rsub|1>|2>|)>,\<b-k\><rsub|3>=\<b-g\><around*|(|t<rsub|n>+<frac|h|2>,\<b-y\><rsub|n>+h*<frac|\<b-k\><rsub|2>|2>|)>>,<math|\<b-k\><rsub|4>=\<b-g\><around*|(|t<rsub|n>+h,\<b-y\><rsub|n>+h\<b-k\><rsub|3>|)>>

  Now the function <math|\<b-g\><around*|(|t,\<b-x\>|)>> is for gradient
  integration

  <math|\<b-g\><around*|(|t,\<b-x\><rsub|>|)>=<frac|\<partial\>\<b-x\><rprime|'><rsub|i>|\<partial\>*t>=<around*|(|\<b-x\><around*|(|t<rsub|>|)>-\<b-x\><rsub|h><around*|(|t<rsub|>|)>|)><rsup|T>\<nabla\><rsub|\<b-w\>>*\<b-f\><around*|(|t,\<b-x\><around*|(|t|)>|)>=\<b-Delta\><around*|(|t<rsub|i>+\<delta\>|)><rsup|T>*\<nabla\><rsub|\<b-w\>>*\<b-f\><around*|(|t,\<b-x\><around*|(|t<rsub|i>+\<delta\>|)>|)>>

  But we cannot compute the modifed <math|t<rsub|i>> values for the error
  term <math|\<b-Delta\><around*|(|t<rsub|i>|)>=<around*|(|\<b-x\><around*|(|t<rsub|i>|)>-\<b-x\><rsub|h><around*|(|t<rsub|i>|)>|)>>.
  To work around this we need to approximate values for
  <math|t=t<rsub|i>+\<delta\>> which are used in Runge-Kutta updates. We do
  linear approximation and have values for <math|t<rsub|i>> and
  <math|t<rsub|i+1>> and intepolate:\ 

  <math|\<b-Delta\><around*|(|t<rsub|i>+\<delta\>|)>=\<b-Delta\><around*|(|t<rsub|i>|)>+<frac|\<delta\>-t<rsub|i>|t<rsub|i+1>-t<rsub|i>>*<around*|(|\<b-Delta\><around*|(|t<rsub|i+1>|)>-\<b-Delta\><around*|(|t<rsub|i>|)>|)>>

  <math|\<b-x\><around*|(|t<rsub|i>+\<delta\>|)>=\<b-x\><around*|(|t<rsub|i>|)>+<frac|\<delta\>-t<rsub|i>|t<rsub|i+1>-t<rsub|i>><around*|(|\<b-x\><around*|(|t<rsub|+1>|)>-\<b-x\><around*|(|t<rsub|i>|)>|)>>\ 

  NOTE: For the last time step we just use value
  <math|\<b-Delta\><around*|(|t<rsub|N>+\<delta\>|)>=\<b-Delta\><around*|(|t<rsub|N>|)>>
  and don't interpolate.\ 

  \;

  <section|Independent Component Analysis and PCA>

  To scale to high dimensions and to find easily predictable signals from
  stock market prices, PCA is used to reduce number of dimensions to
  <math|O<around*|(|10<rsup|2>|)>> signals. Signals are furher processed
  using ICA to find signals that don't have much higher-order correlations.
  The maximum variance signals are used and rest of the signals are dropped.
  To compute inverse of the linear preprocessing transform, pseudoinverse of
  linear matrix is computed.

  ICA learning of 41 days signals show only static prices with no variance in
  data except unpredictable impulse functions (one per signal/day). This
  means there is no much predicatability in data except that you can divide
  stocks to correlating groups with similar price levels or something.

  TODO: Test ICA with 6 months (120 days/samples) data.

  <section|Implementation>

  HMC sampler for learning weights <math|\<b-w\>> from noisy data is
  implemented in <with|font-shape|italic|Dinrhiw2> library. The code is in
  <verbatim|src/neuralnetwork/DiffEQ_HMC.cpp>. In practice, we divide
  historical data to <math|D\<times\>T>-dimensional vector. Vector has
  <math|D> values per observation and have <math|T> time steps for each
  values simulated by differential equations. The first <math|D> observations
  are initial starting point <math|\<b-x\><rsub|0>>.

  <subsection|Testing>

  Code seem to function <with|font-series|bold|quite badly> and requires more
  testing and enabling of bayesian prior for neural network weights which
  disabling did improve results.

  - write testcases learning simple cases and plot the results.

  <em|linear problem x(t)=a*t is learnt correctly, x(t)=sin(f*t) function is
  <with|font-series|bold|NOT> learnt, neural network diff.eq. source
  x(t)=neural_network(x(t)) is <with|font-series|bold|NOT> learn correctly
  but error is reduced a bit. Using same neural network as a source and
  starting point for training.>

  TODO: write bayesian_nnetwork creating code and calculation of predictions
  using bayesian prediction. MAYBE: bayesian treatment can plot N randomly
  chosen diff.eq. models and plot the curves (no useful results).

  =\<gtr\> DON'T WORK NOW EXCEPT IN VERY SIMPLE FUNCTIONS (LINEAR CURVES).
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|1.1|1>>
    <associate|auto-3|<tuple|2|2>>
    <associate|auto-4|<tuple|3|2>>
    <associate|auto-5|<tuple|3.1|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Bayesian
      Neural Network Differential Equation Model>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>Runge-Kutta for
      Jacobian/Gradient <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Independent
      Component Analysis and PCA> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Implementation
      (TODO)> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>