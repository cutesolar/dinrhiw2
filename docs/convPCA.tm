<TeXmacs|2.1>

<style|generic>

<\body>
  <with|font-series|bold|Convolutional PCA> (and ICA)

  <with|font-shape|italic|Novel Insight Research, Tomas Ukkonen, 2022>

  Now that it is possible to calculate fast PCA using superresolutional
  numbers. It is possible to calculate convolutional PCA to find signals that
  are strong in data.

  Let's define <math|\<b-x\><around*|(|n|)>=<around*|[|s<rsub|1><around*|(|n|)>,s<rsub|2><around*|(|n|)>,s<rsub|3><around*|(|n|)>,s<rsub|4><around*|(|n|)>,s<rsub|5><around*|(|n|)>\<ldots\>s<rsub|K><around*|(|n|)>|]><rsup|T>>.
  Calculating this PCA does not allow delays in time, so we further define
  <math|<wide|\<b-x\>|^><around*|(|n|)>=<around*|[|\<b-x\><around*|(|n+k|)>,\<b-x\><around*|(|n+k-1|)>,\<ldots\>\<b-x\><around*|(|n-k|)>|]>>,
  to have maximum of <math|k>-time delay in signals. Signals
  <math|s<rsub|1><around*|(|n|)>> are superresolutional numbers with time
  history of given dimensions of superresolutional numbers (Now <math|d=7>,
  could be <math|d=31>).

  TODO:

  1. Write code stock market data and try to learn convolutional PCA.

  2. Define room with <math|K> microphones (<math|K> dimensional PCA with
  time-delays <math|k> so there are input dimensions <math|2*k*K>). Create
  random audio sources at random locations and calculate measured signals at
  microphone locations. Try to solve inverse PCA to learn source signals
  where learnt signals are convolutions of the measured signals.
  <math|y<rsub|i><around*|(|n|)>=<big|sum><rsub|k><big|sum><rsub|i>a<rsub|i>\<circ\>s<rsub|i><around*|(|n+k|)>>.
  Convolution should allow in audio echo models from the room.

  RESULTS: ConvPCA cannot separate sources reliably(?), estimated signals are
  not sinusoids although sources are sinusoids with different frequencies and
  phases. ConvPCA gives some useful results with
  <math|k=<around*|[|-11,+11|]>>. However, you still need Convolutional
  ICA(?).

  <with|font-series|bold|Convolutional ICA>

  Convolutional ICA need to calculate. FastICA don't seem to work.
  Investigate what are distrbutions of <math|p<around*|(|y|)>:y=w<rsup|T>x >
  when using superreso numbers, are they same with normal distributed
  numbers? Using FastICA to single number dimensions doesn't seem to work.
  Distributions are not same as with normal distributed numbers.

  In FastICA non-linearity <math|G<around*|(|u|)>=-e<rsup|-u<rsup|2>/2>>
  measures non-gaussianity which maximum is at when
  <math|u\<rightarrow\>large>. To map this to superresolutional numbers you
  measure non-gaussianity per component number:
  <math|G<around*|(|s|)>=-<big|prod><rsub|i>G<around*|(|s<rsub|i>|)>=-e<rsup|<rsup|>-0.5<big|sum><rsub|i>s<rsup|2><rsub|i>>=-e<rsup|-0.5*<around*|\<\|\|\>|s|\<\|\|\>><rsup|2><rsub|F>>>.
  Now this derivate per <math|s=\<b-w\><rsup|T>\<b-x\>> is difficult problem.
  We can write superresolutional vector <math|\<b-x\>> as matrix
  <math|\<b-X\>> where components of rows are superresolutional values for
  Frobenius norm.\ 

  <math|<frac|\<partial\>G<around*|(|s=\<b-w\><rsup|T>\<b-x\>|)>|\<partial\>\<b-w\>>=<frac|1|2>e<rsup|-0.5*<around*|\<\|\|\>|s|\<\|\|\>><rsup|2><rsub|F>>*<frac|\<partial\><around*|\<\|\|\>|\<b-w\><rsup|T>\<b-X\>|\<\|\|\>><rsup|2><rsub|F>|\<partial\>\<b-w\>>>,
  <math|<frac|\<partial\><around*|\<\|\|\>|\<b-w\><rsup|T>\<b-X\>|\<\|\|\>><rsup|2><rsub|F>|\<partial\>\<b-w\>>=<frac|\<partial\>tr<around*|(|\<b-w\><rsup|T>\<b-X\>*\<b-X\><rsup|T>*\<b-w\>|)>|\<partial\>\<b-w\>>=**2*\<b-X\>*\<b-X\><rsup|T>*\<b-w\>>

  <math|\<nabla\><rsub|\<b-w\>>G<around*|(|\<b-w\>|)>=<frac|\<partial\>G<around*|(|s=\<b-w\><rsup|T>\<b-x\>|)>|\<partial\>\<b-w\>>=e<rsup|-0.5*<around*|\<\|\|\>|s|\<\|\|\>><rsup|2><rsub|F>>*\<b-X\>*\<b-X\><rsup|T>*\<b-w\>>\ 

  And for FastICA you also need second derivate fo <math|G<around*|(|s|)>> \ 

  <math|H<around*|(|G<around*|(|\<b-w\>|)>|)>=<frac|\<partial\><rsup|2>G<around*|(|s=\<b-w\><rsup|T>\<b-x\>|)>|\<partial\><rsup|2>\<b-w\>>=-e<rsup|-0.5*<around*|\<\|\|\>|s|\<\|\|\>><rsup|2><rsub|F>>*\<b-X\>*\<b-X\><rsup|T>*\<b-w\>*\<b-w\><rsup|T>\<b-X\>*\<b-X\><rsup|T>+e<rsup|-0.5*<around*|\<\|\|\>|s|\<\|\|\>><rsup|2><rsub|F>>*\<b-X\>*\<b-X\><rsup|T>*>

  Now you need to compute expectations to calculation Netwon iteration

  <math|E<rsub|\<b-x\>><around*|{|<frac|\<partial\>G<around*|(|s=\<b-w\><rsup|T>\<b-x\>|)>|\<partial\>\<b-w\>>|}>=E<rsub|\<b-x\>><around*|{|e<rsup|-0.5*<around*|\<\|\|\>|s|\<\|\|\>><rsup|2><rsub|F>>*\<b-X\>*\<b-X\><rsup|T>*\<b-w\>|}>>,(
  this can be computed directory from the data.

  <math|E<rsub|\<b-x\>><around*|{|<frac|\<partial\><rsup|2>G<around*|(|s=\<b-w\><rsup|T>\<b-x\>|)>|\<partial\><rsup|2>\<b-w\>>|}>=E<rsub|\<b-x\>><around*|{|-e<rsup|-0.5*<around*|\<\|\|\>|s|\<\|\|\>><rsup|2><rsub|F>>*\<b-X\>*\<b-X\><rsup|T>*\<b-w\>*\<b-w\><rsup|T>\<b-X\>*\<b-X\><rsup|T>+e<rsup|-0.5*<around*|\<\|\|\>|s|\<\|\|\>><rsup|2><rsub|F>>*\<b-X\>*\<b-X\><rsup|T>|}>>,
  Now <math|E<rsub|\<b-x\>><around*|{|\<b-X\>*\<b-X\><rsup|T>|}>\<approx\>\<b-I\>>
  because superresolutional numbers are white
  <math|E<around*|[|s<rsub|i*>s<rsub|j>|]>=\<delta\><around*|(|i-j|)>>
  [assuming circular convolution is a bit like vector multiplication where
  high dimensions are small so it's regular convolution <math|\<Rightarrow\>>
  linear]. This may mean we may take approximations

  <math|E<rsub|\<b-x\>><around*|{|<frac|\<partial\>G<around*|(|s=\<b-w\><rsup|T>\<b-x\>|)>|\<partial\>\<b-w\>>|}>\<approx\>E<rsub|\<b-x\>><around*|{|e<rsup|-0.5*<around*|\<\|\|\>|s|\<\|\|\>><rsup|2><rsub|F>>**\<b-w\>|}>>

  <math|E<rsub|\<b-x\>><around*|{|<frac|\<partial\><rsup|2>G<around*|(|s=\<b-w\><rsup|T>\<b-x\>|)>|\<partial\><rsup|2>\<b-w\>>|}>=E<rsub|\<b-x\>><around*|{|e<rsup|-0.5*<around*|\<\|\|\>|s|\<\|\|\>><rsup|2><rsub|F>>*<around*|(|\<b-I\>-\<b-w\>*\<b-w\><rsup|T>|)>|}>>

  Now we need Newton iteration step to solve zero point for gradient
  <math|G<around*|(|s=\<b-w\><rsup|T>\<b-x\>|)>> function:

  <\math>
    \<b-w\><rsub|t+1>=\<b-w\><rsub|t>-<around*|(|E<rsub|\<b-x\>><around*|{|e<rsup|+0.5*<around*|\<\|\|\>|s|\<\|\|\>><rsup|2><rsub|F>>*|}><around*|(|\<b-I\>-\<b-w\>*<rsub|t>\<b-w\><rsub|t><rsup|T>|)>|)><rsup|-1>*E<rsub|\<b-x\>><around*|{|e<rsup|-0.5*<around*|\<\|\|\>|s|\<\|\|\>><rsup|2><rsub|F>>**|}>\<b-w\><rsub|t>

    \<b-w\><rsub|t+1>=\<b-w\><rsub|t>-**<around*|(|<around*|(|\<b-I\>-\<b-w\>*<rsub|t>\<b-w\><rsub|t><rsup|T>|)>|)><rsup|-1>*\<b-w\><rsub|t>
  </math>

  But this iteration does NOT depend on data so the approximation must be
  wrong!

  Now, instead we don't take approximation
  <math|E<rsub|\<b-x\>><around*|{|\<b-X\>*\<b-X\><rsup|T>|}>=\<b-I\>>. In
  this case we must solve whole Hessian matrix from the data, and hope it is
  well-defined and calculate it's inverse.

  <math|\<b-w\><rsub|t+1>=\<b-w\><rsub|t>-H<around*|(|\<b-w\><rsub|t>|)><rsup|-1>*\<nabla\>G<around*|(|\<b-w\><rsub|t>|)>>

  \;

  \;

  =====

  Compare this with Linear ICA where input data is data with added
  time-delays <math|<wide|\<b-x\>|^><around*|(|n|)>>. Try to solve demixing
  error from a room with random mic and audio signal sources. Another problem
  is Brain EEG measurements where measurement points are fixed near brain and
  you learn deconvolutional sources from brain. With Interaxon Muse you have
  only 4 measurement points so you get 4 signals.

  Research problem: how to do convolutional ICA, you get signals with finite
  time horizon <math|s<rsub|i><around*|(|n|)>> and You need to do convolution
  <math|b\<circ\>s<rsub|i><around*|(|n|)>> so that you maximize
  non-gaussianity/kurtosis or something like that given finite time horizon
  (single variable <math|s<rsub|i><around*|(|n|)>>) and multiple samples
  <math|<around*|{|s<rsub|i><around*|(|n-k|)>|}><rsub|k>>.

  \;

  Tomas Ukkonen, M.Sc.
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>