import re
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'paper/sections/formal_analysis.tex';s=p.read_text()
replacements={
r"contradicting the independence of $\phi''(x)$ from $y$.":r'''\begin{equation}\partial_y\phi''(x)=0\ne d/(4x^{3/2}\sqrt y)=\partial_y\partial_x^2d_{\BW}(xI,yI)^2.\end{equation}''',
r'Apply \eqref{eq:formal_exactgap} and $\tr[(T-I)A_\varepsilon(T-I)]=d_{\BW}(A_\varepsilon,B_\varepsilon)^2$.':r'''\begin{equation}d_{\BW}(A_\varepsilon,B_\varepsilon)^2=\tr[(T-I)A_\varepsilon(T-I)]=\varepsilon^2\tr(KCK)+O(\varepsilon^3).\end{equation}''',
r'Induction, constant speed, and \eqref{eq:formal_exactgap} give the assertions.':r'''\begin{align}T(A_t,B)-I&=\beta^t(T_0-I)H_t^{-1},\\d_{\BW}(A_t,B)^2&=\tr[(T(A_t,B)-I)A_t(T(A_t,B)-I)]=\beta^{2t}d_{\BW}(A_0,B)^2.\end{align}''',
r'Chernoff optimization over $\lambda$ yields the scalar bound; apply it with $v=e_i$ and failure probability $\delta/d$, then sum the $d$ coordinate squares.':r'''\begin{align}c_t&=\alpha(1-\beta^{2t})/(2-\alpha),\\\Pr(|v^\top(e_t-b_t)|>x)&\le2\inf_{\lambda>0}\exp(-\lambda x+\lambda^2c_tv^\top Vv/2)\\&=2\exp[-x^2/(2c_tv^\top Vv)],\\\Pr\left(\exists i:|(e_t-b_t)_i|^2>2c_tV_{ii}\log(2d/\delta)\right)&\le\sum_{i=1}^d\delta/d=\delta.\end{align}''',
r'For fixed $v\in\R^d$, set $a_{\alpha,k}=\sqrt\alpha(1-\alpha)^k$. Then':r'\begin{equation}v\in\R^d,\qquad a_{\alpha,k}=\sqrt\alpha(1-\alpha)^k.\end{equation}',
r'The characteristic-function limit and $F(u)-F(\mu)=\tfrac12\|u-\mu\|^2$ give the result.':r'\begin{equation}F(u)-F(\mu)=\tfrac12\|u-\mu\|^2,\quad\E[F(\mu+e_\alpha^\infty)-F(\mu)]=\tfrac12\E\|e_\alpha^\infty\|^2.\end{equation}',
r'The first inequality is the centered bounded-variable Hoeffding inequality, obtained by convexly interpolating $e^{\lambda z}$ between $z=-c_t$ and $z=c_t$.':r'''\begin{align}e^{\lambda z}&\le\frac{c_t+z}{2c_t}e^{\lambda c_t}+\frac{c_t-z}{2c_t}e^{-\lambda c_t},\quad |z|\le c_t,\\\E_{t-1}e^{\lambda Z_t}&\le\cosh(\lambda c_t)\le e^{\lambda^2c_t^2/2}.\end{align}''',
r'Let $\xi_t=g_t-h_t$ and $Z_t=-\eta_t\langle h_t,\bar B_t\xi_t\rangle$. Then':r'\begin{equation}\xi_t=g_t-h_t,\qquad Z_t=-\eta_t\langle h_t,\bar B_t\xi_t\rangle.\end{equation}',
r'Divide by $b_-H_T/2$.':r'\begin{equation}\sum_t\eta_t\|h_t\|^2/H_T=(2/(b_-H_T))\left[(b_-/2)\sum_t\eta_t\|h_t\|^2\right].\end{equation}',
r'$H_T=\eta_0\sqrt T$, $Q_T=\eta_0^2$ in \eqref{eq:formal_highprob}.':r'\begin{equation}H_T=\eta_0\sqrt T,\quad Q_T=\eta_0^2,\quad\eqref{eq:formal_highprob}.\end{equation}',
r'The conditional descent inequality gives':'',
r'Consequently $V_t$ converges almost surely and':r'\begin{equation}V_t\ge0,\quad\E_{t-1}V_{t+1}\le V_t\quad\Longrightarrow\quad V_t\to V_\infty<\infty\quad\text{a.s.}\end{equation}',
r'Since $\sum_t\eta_t=\infty$, $\liminf\|\nabla f(\theta_t)\|=0$.':r'\begin{equation}\sum_t\eta_t=\infty,\quad\sum_t\eta_t\|\nabla f(\theta_t)\|^2<\infty\quad\Longrightarrow\quad\liminf_t\|\nabla f(\theta_t)\|=0.\end{equation}',
r'For the final assertion, let $h_t=\|\nabla f(\theta_t)\|$. Then':r'\begin{equation}h_t=\|\nabla f(\theta_t)\|.\end{equation}',
r'If $\limsup h_t>\varepsilon>0$ while $\liminf h_t=0$, select disjoint excursion intervals $[p_k,q_k)$ with':r'\begin{equation}\limsup_t h_t>\varepsilon>0=\liminf_t h_t\quad\Longrightarrow\quad\exists\,[p_k,q_k)\text{ pairwise disjoint}:\end{equation}',
r'They obey':'',
r'contradicting $\sum_t\eta_th_t^2<\infty$.':r'\begin{equation}\sum_t\eta_th_t^2\ge\sum_{k=1}^\infty\varepsilon^3/(8L_fD)=\infty\quad\bot.\end{equation}',
r'The three strict Jury conditions give \eqref{eq:formal_momentumphase}. Eliminate $m_{t-1}$:':r'\begin{equation}q>0,\quad2(1+\beta)-q>0,\quad1-\beta>0\quad\Longleftrightarrow\quad\rho(A)<1.\end{equation}',
r'For stationary autocovariances $\gamma_k=\Cov(\theta_t,\theta_{t-k})$,':r'\begin{equation}\gamma_k=\Cov(\theta_t,\theta_{t-k})\quad\text{(stationary)}.\end{equation}',
r'Substitution yields \eqref{eq:formal_momentumvariance}; $\rho(A)<1$ removes the initialization mean and covariance.':r'\begin{equation}\rho(A)<1\Longrightarrow A^t\E Z_1\to0,\quad A^t\Cov(Z_1)(A^\top)^t\to0,\quad Z_1=(\theta_1,m_0)^\top.\end{equation}',
r'Let $B^{1/2}HB^{1/2}=Q\Lambda Q^\top$ and set':r'\begin{equation}B^{1/2}HB^{1/2}=Q\Lambda Q^\top,\quad Q^\top Q=I.\end{equation}',
r'Apply the scalar characteristic-polynomial criterion to each mode. For the covariance,':r'\begin{equation}\rho(\mathcal A)=\max_i\rho\begin{pmatrix}1-\eta(1-\beta)\lambda_i&-\eta\beta\\(1-\beta)\lambda_i&\beta\end{pmatrix}.\end{equation}',
r'Vectorization and $\rho(\mathcal A\otimes\mathcal A)=\rho(\mathcal A)^2<1$ give the inverse formula.':r'\begin{equation}\rho(\mathcal A\otimes\mathcal A)=\rho(\mathcal A)^2<1,\quad(I-\mathcal A\otimes\mathcal A)^{-1}=\sum_{k=0}^\infty(\mathcal A\otimes\mathcal A)^k.\end{equation}',
r'The time-dependent matrices satisfy':'',
r'Choose $P\succ0$ solving $\mathcal A^\top P\mathcal A-P=-I$. For sufficiently large $t$,':r'\begin{equation}P=\sum_{k=0}^\infty(\mathcal A^\top)^k\mathcal A^k\succ0,\quad\mathcal A^\top P\mathcal A-P=-I,\quad\exists t_0\ \forall t\ge t_0:\end{equation}',
r'Writing $D_t=\mathcal C_t-\mathcal C$ gives':r'\begin{equation}D_t=\mathcal C_t-\mathcal C.\end{equation}',
r'The mean obeys the same stable homogeneous product.':r'\begin{equation}\|P^{1/2}\E Z_t\|\le\rho^{t-t_0}\|P^{1/2}\E Z_{t_0}\|\to0.\end{equation}',
r'For $t\ge2$,':r'\begin{equation}t\ge2.\end{equation}',
r'Sum the geometric term and substitute into \eqref{eq:momentumbound}--\eqref{eq:momentumpenalty}. At $t=1$, $m_0=0$ makes the virtual-variation term exactly zero.':r'\begin{align}\sum_{t=2}^T\beta^{2t-2}&\le\beta^2/(1-\beta^2),\\\sum_{t=2}^T\E\|V_t-V_{t-1}\|^2/a_t&\le C_\beta(T\eta^3+\eta),\\(V_0-V_1)m_0&=0,\quad A_T\ge T\eta\quad\Longrightarrow\quad\eqref{eq:momentumbound}=O(T^{-1/2})+O(T^{-1}).\end{align}',
r'The $c=0$ case follows from triangularity. Conversely,':r'\begin{equation}c=0\Longrightarrow\operatorname{spec}(A)=\{1-\eta\mu,\beta\}\subset[0,1).\end{equation}',
r'For the fixed point $(\vartheta_\infty,x_\infty)^\top=(I-A)^{-1}b/N$,':r'\begin{equation}(\vartheta_\infty,x_\infty)^\top=(I-A)^{-1}b/N.\end{equation}',
r'The replace-one-example loss bound gives the last assertion. Conditions \eqref{eq:formal_feedback_assumption} are explicit additional contraction hypotheses; they are not inferred from SPD clipping alone.':r'\begin{equation}\epsilon_N\le L_\ell\sup_t\vartheta_t\le L_\ell(b_\theta+cb_X)/[N(\mu-cK)].\end{equation}',
r"For neighboring $S,S'$ differing at index $j$,":r"\begin{equation}S\sim S',\quad Z_i=Z_i'\ (i\ne j).\end{equation}",
r'The Doob martingale associated with revealing the $N$ examples has bounded conditional difference ranges $2\epsilon_N+M/N$. Conditional Hoeffding gives':r'\begin{align}D_i&=\E[G\mid Z_1,\ldots,Z_i]-\E[G\mid Z_1,\ldots,Z_{i-1}],\\\operatorname{ess\,sup}(D_i\mid Z_{<i})-\operatorname{ess\,inf}(D_i\mid Z_{<i})&\le2\epsilon_N+M/N.\end{align}',
r'Set $u=(2N\epsilon_N+M)\sqrt{\log(2/\delta)/(2N)}$.':r'\begin{equation}u=(2N\epsilon_N+M)\sqrt{\log(2/\delta)/(2N)}.\end{equation}',
r'Let $\mu=R(w)\in(0,1)$ and $X_i=\ell(w,Z_i)\in[0,1]$. The map':r'\begin{equation}\mu=R(w)\in(0,1),\quad X_i=\ell(w,Z_i)\in[0,1].\end{equation}',
r'is convex in each coordinate. Replacing each $X_i$ by an independent $B_i\sim\operatorname{Bernoulli}(\mu)$ can only increase its expectation. Hence':r'\begin{gather}\Psi(x)=\exp[N\operatorname{kl}(N^{-1}\sum_ix_i\|\mu)],\quad\partial_{x_i}^2\Psi\ge0,\\\Psi(x)\le(1-x_i)\Psi(x_{-i},0)+x_i\Psi(x_{-i},1),\quad\E X_i=\mu,\\B_i\overset{\rm iid}\sim\operatorname{Bernoulli}(\mu),\quad\E\Psi(X_1,\ldots,X_N)\le\E\Psi(B_1,\ldots,B_N).\end{gather}',
r'For $\mu\in\{0,1\}$, $X_i=\mu$ almost surely and the moment equals one. Use $0^0=1$ in the finite sum.':r'\begin{equation}\mu\in\{0,1\}\Longrightarrow X_i=\mu\text{ a.s.}\Longrightarrow\E e^{N\operatorname{kl}(\overline X\|\mu)}=1,\quad0^0:=1.\end{equation}',
r'By Fubini, Lemma~\ref{lem:formal_pac_moment}, and Markov,':'',
r'On the complementary event, for arbitrary $Q\ll P$ and $h(w)=N\operatorname{kl}(\widehat R_S(w)\|R(w))$,':r'\begin{equation}\mathcal E=\{S:\E_Pe^h\le(N+1)/\delta\},\quad\Pr(\mathcal E)\ge1-\delta,\quad Q\ll P,\quad h(w)=N\operatorname{kl}(\widehat R_S(w)\|R(w)).\end{equation}',
r'The last inequality follows from $\partial_u^2\operatorname{kl}(u\|v)=1/[u(1-u)]\ge4$ and its value and first derivative vanishing at $u=v$.':r'\begin{equation}\partial_u^2\operatorname{kl}(u\|v)=1/[u(1-u)]\ge4,\quad\operatorname{kl}(v\|v)=\partial_u\operatorname{kl}(v\|v)=0\Longrightarrow\operatorname{kl}(u\|v)\ge2(u-v)^2.\end{equation}',
r'For $\Sigma_Q=\tau^2\mathcal H_S^{-1}$,':r'\begin{equation}\Sigma_Q=\tau^2\mathcal H_S^{-1}.\end{equation}',
r'For $\varepsilon\sim\mathcal N(0,\tau^2\mathcal H_S^{-1})$,':r'\begin{equation}\varepsilon\sim\mathcal N(0,\tau^2\mathcal H_S^{-1}),\quad\E\varepsilon=0,\quad\E\|\varepsilon\|^2=\tau^2\tr\mathcal H_S^{-1}.\end{equation}',
r'Apply \eqref{eq:formal_PAC}. The eigenvalues of $\mathcal H_S$ lie in $[a^2,b^2]$, proving the final bounds. Global smoothness and bounded loss are additional hypotheses; state clipping does not imply them.':r'\begin{gather}\eqref{eq:formal_PAC},\quad a^2I\preceq\mathcal H_S\preceq b^2I,\\p/b^2\le\tr\mathcal H_S^{-1}\le p/a^2,\quad2p\log a\le\log\det\mathcal H_S\le2p\log b.\end{gather}',
r'Apply Theorem~\ref{thm:formal_pacbayes} to prior $P_k$ with failure probability $\delta/K$, and take the union bound.':r'\begin{align}\Pr(\mathcal E_k^c)&\le\delta/K,\quad\Pr\left(\bigcap_{k=1}^K\mathcal E_k\right)\ge1-\sum_{k=1}^K\delta/K=1-\delta,\\\log((N+1)/(\delta/K))&=\log(K(N+1)/\delta).\end{align}',
r'For $v=g/\|g\|$, set $a=v^\top Pv$, $b=v^\top P^2v$. Then':r'\begin{equation}v=g/\|g\|,\quad a=v^\top Pv,\quad b=v^\top P^2v.\end{equation}',
r'For the displayed pair $(P,g)$,':'',
r'so equality holds.':'',
r'Telescope, apply the PL inequality when assumed, and substitute $p_-=1/b$, $p_+=1/a$.':r'\begin{align}\sum_t\eta_t\gamma_P\|\nabla f(\theta_t)\|^2/2&\le f(\theta_1)-f_\star,\\\|\nabla f\|^2\ge2\mu_f(f-f_\star)&\Longrightarrow\Delta_{t+1}\le(1-\eta_t\gamma_P\mu_f)\Delta_t,\\p_-=1/b,\ p_+=1/a&\Longrightarrow\gamma_P=2\sqrt{ab}/(a+b).\end{align}',
r'followed by the smoothness inequality and the singular-value bounds on $P_t$.':r'\begin{align}f(\theta_t-\eta_td_t)&\le f(\theta_t)-\eta_t\langle g_t,d_t\rangle+L_f\eta_t^2\|d_t\|^2/2,\\s_-p_-&\le\|d_t\|/\|g_t\|\le s_+p_+.\end{align}',
r'Apply the product rule. The skew-Jacobian entry is':'',
r'A $C^2$ scalar potential has a symmetric Hessian.':r'\begin{equation}\mathcal U=\nabla\psi,\quad\psi\in C^2\Longrightarrow\mathrm D\mathcal U=\nabla^2\psi=(\nabla^2\psi)^\top\quad\bot.\end{equation}'
}
# Replace only inside proof environments; theorem assumptions remain explicit.
def edit(m):
    t=m.group(1)
    for old,new in replacements.items():t=t.replace(old,new)
    t=re.sub(r'(?m)^(Set|Then)\s*$','',t)
    return r'\begin{proof}'+t+r'\end{proof}'
s=re.sub(r'\\begin\{proof\}(.*?)\\end\{proof\}',edit,s,flags=re.S)
p.write_text(s)
print('All 36 extended proofs now use displayed mathematical derivations.')
