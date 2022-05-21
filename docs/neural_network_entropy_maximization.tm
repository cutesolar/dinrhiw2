<TeXmacs|2.1>

<style|generic>

<\body>
  <with|font-series|bold|Entropy maximization of neural network ouputs and
  entropy regularization of reinforcement learning>\ 

  Tomas Ukkonen, <with|font-shape|italic|Novel Insight>,
  <verbatim|tomas.ukkonen@novelinsight.fi>, 2021

  <with|font-series|bold|1. Entropy maximization>

  Reinforcement learning requires maximization of entropy
  <math|H<around*|(|\<b-Y\>|)>> of outputs so there is enough exploration. In
  practise MSE minimized outputs can be tranformed to entropy by using
  formula:

  <\padded-center>
    <math|H<around*|(|\<b-Y\><around*|\||\<b-w\>|\<nobracket\>>|)>=-<big|sum><rsub|i><frac|exp<around*|(|\<alpha\>*\<b-y\><around*|(|i<around*|\||\<b-w\>|\<nobracket\>>|)>|)>|<big|sum><rsub|j>exp<around*|(|\<alpha\>*\<b-y\><around*|(|j<around*|\||\<b-w\>|\<nobracket\>>|)>|)>>*log<around*|(|<frac|exp<around*|(|\<alpha\>*\<b-y\><around*|(|i<around*|\||\<b-w\>|\<nobracket\>>|)>|)>|<big|sum><rsub|j>exp<around*|(|\<alpha\>*\<b-y\><around*|(|j<around*|\||\<b-w\>|\<nobracket\>>|)>|)>>*|)>>.
  </padded-center>

  In practice we want to maximize <math|H<around*|(|\<b-Y\><around*|\||\<b-w\>|\<nobracket\>>|)>>
  and calculate its gradient. First we calculate gradient of entropy function
  <math|H<around*|(|\<b-Y\>|)>=-<big|sum><rsub|i>*p<rsub|i>*log<around*|(|p<rsub|i>|)>>
  and we get <math|<frac|\<partial\>H<around*|(|\<b-Y\>|)>|\<partial\>\<b-w\>>=-<big|sum><rsub|i><frac|\<partial\>p<rsub|i>|\<partial\>\<b-w\>><around*|(|1+log<around*|(|p<rsub|i>|)>|)>>.
  Next, we must calculate the <math|<frac|\<partial\>p<rsub|i>|\<partial\>\<b-w\>>>
  term:

  <\padded-center>
    <\math>
      <frac|\<partial\>p<rsub|i>|\<partial\>\<b-w\>>=<frac|\<partial\><rsub|>|\<partial\>\<b-w\>>*<frac|exp<around*|(|\<alpha\>*\<b-y\><around*|(|i<around*|\||\<b-w\>|\<nobracket\>>|)>|)>|<big|sum><rsub|j>exp<around*|(|\<alpha\>*\<b-y\><around*|(|j<around*|\||\<b-w\>|\<nobracket\>>|)>|)>>

      =

      <around*|(|\<alpha\>*exp<around*|(|\<alpha\>*y<around*|(|i|)>|)>*<frac|\<partial\><rsub|>y<around*|(|i|)>|\<partial\>\<b-w\>><around*|(|<big|sum><rsub|j>exp<around*|(|\<alpha\>*\<b-y\><around*|(|j<around*|\||\<b-w\>|\<nobracket\>>|)>|)>|)>-\<alpha\>exp<around*|(|\<alpha\>*y<around*|(|i|)>|)><big|sum><rsub|j>exp<around*|(|\<alpha\>*y<around*|(|j|)>|)><frac|\<partial\><rsub|>y<around*|(|j|)>|\<partial\>\<b-w\>>|)>/<around*|(|<big|sum><rsub|j>exp<around*|(|\<alpha\>*\<b-y\><around*|(|j<around*|\||\<b-w\>|\<nobracket\>>|)>|)>|)><rsup|2>
    </math>
  </padded-center>

  This calculates entropy gradient of the neural network output. It can be
  calculated from the jacobian matrix <math|<frac|\<partial\>\<b-y\>|\<partial\>\<b-w\>>>
  given scaling factor alpha <math|\<alpha\>> or simply decide
  <math|\<alpha\>> to be one. This assumes outputs from the neural network
  are logits of probability values.

  \;

  <with|font-series|bold|2. Entropy regularization of reinforcement learning>

  In practice, we assume error function is now softmax function of Q-values
  (linear output layer) and the target is also probability distribution of
  actions and we calculate Kullback-Leibler divergence between those values.

  <\padded-center>
    <math|D<rsub|K*L><around*|(|<with|font-series|bold|Q><rsub|target><around*|\|||\|>\<b-P\>|)><around*|[|\<b-w\>|]>=-<big|sum><rsub|i>q<around*|(|i|)>*log<around*|(|<frac|p<around*|(|i<around*|\||\<b-w\>|\<nobracket\>>|)>|q<around*|(|i|)>>|)>=<big|sum><rsub|i>q<around*|(|i|)><around*|(|log<around*|(|q<around*|(|i|)>-log<around*|(|p<around*|(|i<around*|\||\<b-w\>|\<nobracket\>>|)>|)>|)>|\<nobracket\>>*>
  </padded-center>

  We take the gradient of <math|D<rsub|KL>> and get
  <math|\<nabla\><rsub|\<b-w\>>D<rsub|K*L>=-<big|sum><rsub|i><around*|[|q<around*|(|i|)>/p<around*|(|i<around*|\||\<b-w\>|\<nobracket\>>|)>|]>\<nabla\><rsub|\<b-w\>>*p<around*|(|i<around*|\||\<b-w\>|\<nobracket\>>|)>>.
  For zero <math|p<around*|(|i|)>> values we add small positive epsilon value
  to probabilities to avoid division by zero. Next we must calculate gradient
  of softmax output of the neural network which we computed in the previous
  section. The result is:\ 

  <\padded-center>
    <\math>
      <frac|\<partial\>p<rsub|i>|\<partial\>\<b-w\>>=<frac|\<partial\><rsub|>|\<partial\>\<b-w\>>*<frac|exp<around*|(|*\<b-y\><around*|(|i<around*|\||\<b-w\>|\<nobracket\>>|)>|)>|<big|sum><rsub|j>exp<around*|(|*\<b-y\><around*|(|j<around*|\||\<b-w\>|\<nobracket\>>|)>|)>>

      =

      <around*|(|exp<around*|(|y<around*|(|i|)>|)>*<frac|\<partial\><rsub|>y<around*|(|i|)>|\<partial\>\<b-w\>><around*|(|<big|sum><rsub|j>exp<around*|(|*\<b-y\><around*|(|j<around*|\||\<b-w\>|\<nobracket\>>|)>|)>|)>-exp<around*|(|*y<around*|(|i|)>|)><big|sum><rsub|j>exp<around*|(|*y<around*|(|j|)>|)><frac|\<partial\><rsub|>y<around*|(|j|)>|\<partial\>\<b-w\>>|)>/<around*|(|<big|sum><rsub|j>exp<around*|(|*\<b-y\><around*|(|j<around*|\||\<b-w\>|\<nobracket\>>|)>|)>|)><rsup|2>

      =exp<around*|(|y<rsub|i>|)><around*|(|T<around*|(|\<b-w\>|)>*<frac|\<partial\><rsub|>y<rsub|i>|\<partial\>\<b-w\>>*-<big|sum><rsub|j>exp<around*|(|*y<rsub|j>|)><frac|\<partial\><rsub|>y<rsub|j>|\<partial\>\<b-w\>>|)>/T<around*|(|\<b-w\>|)><rsup|2>
    </math>

    =<math|<around*|(|exp<around*|(|y<rsub|i>|)>/T<around*|(|\<b-w\>|)><rsup|2>|)><around*|(|<big|sum><rsub|j><around*|(|T<around*|(|\<b-w\>|)>\<delta\><around*|(|i-j|)>-exp<around*|(|*y<rsub|j>|)>|)><frac|\<partial\><rsub|>y<rsub|j>|\<partial\>\<b-w\>>|)>>

    <math|=s<around*|(|\<b-w\>,i,\<b-y\>|)><rsup|T>><math|<frac|\<partial\><rsub|>\<b-y\>|\<partial\>\<b-w\>>>\ 

    <math|s<around*|(|\<b-w\>,i,\<b-y\>|)><around*|[|j|]>=<around*|(|exp<around*|(|y<rsub|i>|)>/T<around*|(|\<b-w\>|)><rsup|2>|)>*<around*|(|T<around*|(|\<b-w\>|)>\<delta\><around*|(|i-j|)>-exp<around*|(|*y<rsub|j>|)>|)>>
  </padded-center>

  <\padded-center>
    <math|<frac|\<partial\>\<b-p\>|\<partial\>\<b-w\>>=S<around*|(|\<b-w\>,\<b-y\>|)><rsup|T><frac|\<partial\><rsub|>\<b-y\>|\<partial\>\<b-w\>>>
  </padded-center>

  This last formula can be used as a basis for backpropagation. However, this
  is not fully usable yet because we must also use Kullback-Leible divergence
  formula which is

  <\padded-center>
    <math|\<nabla\><rsub|\<b-w\>>D<rsub|K*L>=-<big|sum><rsub|i><around*|[|q<rsub|i>/p<rsub|i><around*|(|\<b-w\>|)>|]>\<nabla\><rsub|\<b-w\>>*p<rsub|i><around*|(|\<b-w\>|)>=\<b-e\><around*|(|\<b-w\>|)><rsup|T><frac|\<partial\>\<b-p\>|\<partial\>\<b-w\>>=\<b-e\><around*|(|\<b-w\>|)><rsup|T>S<around*|(|\<b-w\>,\<b-y\>|)><rsup|T><frac|\<partial\><rsub|>\<b-y\>|\<partial\>\<b-w\>>=\<b-d\><around*|(|\<b-w\>,\<b-y\>|)><rsup|T><frac|\<partial\><rsub|>\<b-y\>|\<partial\>\<b-w\>>>
  </padded-center>

  This result gives the last layer error vector for backpropagation which is
  <math|\<b-d\><around*|(|\<b-w\>,\<b-y\>|)>=\<b-S\><around*|(|\<b-w\>,\<b-y\>|)>*\<b-e\><around*|(|\<b-w\>|)>>
  meaning we don't have to calculate whole jacobian matrix for calculating
  gradient.

  This similar method can be also used to calculate gradient of entropy
  <math|H<around*|(|\<b-Y\><around*|\||\<b-w\>|\<nobracket\>>|)>> of the
  output which is used to increase randomness of the selected function. In
  practise it is:\ 

  <\padded-center>
    <math|\<nabla\><rsub|\<b-w\>>*H<around*|(|\<b-Y\>|)>=-<big|sum><rsub|i><frac|\<partial\>p<rsub|i>|\<partial\>\<b-w\>><around*|(|1+log<around*|(|p<rsub|i>|)>|)>=\<b-h\><around*|(|\<b-w\>|)><rsup|T><frac|\<partial\>\<b-p\>|\<partial\>\<b-w\>>=\<b-h\><around*|(|\<b-w\>|)><rsup|T>S<around*|(|\<b-w\>,\<b-y\>|)><rsup|T><frac|\<partial\><rsub|>\<b-y\>|\<partial\>\<b-w\>>>
  </padded-center>

  In the discrete reinforcement learning step we use
  <math|\<b-Q\><rsub|new><around*|(|s,\<b-a\>|)>> as normally and calculate
  <math|\<b-q\>=softmax<around*|(|\<b-Q\><rsub|new>|)>> to create a target
  values for Kullback-Leibler divergence based distance optimization to
  change action selection probabilities. Additionally we add the entropy term
  so we are minimizing recurrent neural network with the following error
  function:

  <\padded-center>
    <math|e<around*|(|\<b-w\>|)>=<frac|1|N><big|sum><rsub|i>D<rsub|K*L><around*|(|softmax<around*|(|\<b-Q\><rsub|new,i>|)><around*|\|||\|>P<around*|(|\<b-w\>|)>|)>-\<alpha\>*H<around*|(|\<b-Y\><around*|\||\<b-w\>|\<nobracket\>>|)>>
  </padded-center>

  And we select <math|\<alpha\>> term to be quite small, maybe
  <math|\<alpha\>=0.01>.

  \;

  <with|font-series|bold|Reverse Kullback-Leibler divergence (Better for
  reinforcement learning)>

  Reverse KL-divergence should work better with reinforcement learning(???)
  so we derive reverse KL divergence <math|>

  <\padded-center>
    <math|D<rsub|K*L><around*|(|<with|font-series|bold|\<b-P\><rsub|\<b-w\>>><around*|\|||\|>\<b-Q\><rsub|target>|)><around*|[|\<b-w\>|]>=-<big|sum><rsub|i>p<around*|(|i<around*|\||\<b-w\>|\<nobracket\>>|)>*log<around*|(|<frac|q<around*|(|i|)>|p<around*|(|i<around*|\||\<b-w\>|\<nobracket\>>|)>>|)>=<big|sum><rsub|i>p<around*|(|i<around*|\||\<b-w\>|\<nobracket\>>|)><around*|(|log<around*|(|p<around*|(|i<around*|\||\<b-w\>|\<nobracket\>>|)>|)>-log<around*|(|q<around*|(|i|)>|)>|\<nobracket\>>*>
  </padded-center>

  And the gradient is:

  <\padded-center>
    <math|\<nabla\><rsub|\<b-w\>>D<rsub|K*L>=-<big|sum><rsub|i><around*|[|1+log<around*|(|p<rsub|i><around*|(|\<b-w\>|)>|)>-log<around*|(|q<rsub|i>|)>|]>\<nabla\><rsub|\<b-w\>>*p<rsub|i><around*|(|\<b-w\>|)>=\<b-e\><around*|(|\<b-w\>|)><rsup|T><frac|\<partial\>\<b-p\>|\<partial\>\<b-w\>>=\<b-f\><around*|(|\<b-w\>|)><rsup|T>S<around*|(|\<b-w\>,\<b-y\>|)><rsup|T><frac|\<partial\><rsub|>\<b-y\>|\<partial\>\<b-w\>>=\<b-d\><around*|(|\<b-w\>,\<b-y\>|)><rsup|T><frac|\<partial\><rsub|>\<b-y\>|\<partial\>\<b-w\>>>
  </padded-center>

  So the implementation of the reverse gradient is easy.
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>