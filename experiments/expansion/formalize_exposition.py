"""Keep mathematical statements; replace external exposition with explicit definitions and scope conditions."""
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];p=ROOT/'paper/sections/theory_proofs.tex';s=p.read_text()
# Existing scientific graphics and measurement table belong with experimental details.
moved=[]
def move(m):moved.append(m.group());return ''
s=re.sub(r'\\begin\{(figure|table\*?)\}.*?\\end\{\1\}',move,s,flags=re.S)
(ROOT/'paper/sections/theory_measurements.tex').write_text('\n'.join(moved)+'\n')
first=s.index(r'\subsection{Geometry from')
s=r'''\section{Complete Theory and Proofs}
\label{app:proofs}
\subsection{Probability space, objectives, and stored states}
\begin{align}
\theta&\in\R^p,\quad S=(Z_1,\ldots,Z_N)\sim\mathcal D^N,\quad
f(\theta)=\E_Z\ell(\theta;Z),\quad f_S(\theta)=N^{-1}\sum_i\ell(\theta;Z_i),\\
\mathcal F_t&=\sigma(S,\theta_1,I_1,\omega_1,\ldots,I_t,\omega_t),\quad
\E_t[\cdot]=\E[\cdot\mid\mathcal F_t],\\
\theta_t,m_{t-1},X_{t-1},K_{t-1}&\in\mathcal F_{t-1},\quad g_t,X_t,s_t\in\mathcal F_t,\\
\E[g_t\mid\mathcal F_{t-1}]&=\nabla f(\theta_t),\quad
\E[\|g_t-\nabla f(\theta_t)\|^2\mid\mathcal F_{t-1}]\le\sigma_t^2.
\label{eq:oracle}
\end{align}
\begin{align}
\mathcal K_d(a,b)&=\{A\in\mathbb S^d:aI\preceq A\preceq bI\},\quad0<a\le b<\infty,\\
\Pi_{[a,b]}(A)&=Q\diag(\min\{b,\max\{a,\lambda_i(A)\}\})Q^\top,\\
X&=(L,R),\quad P(X)=R^{-1/2}\otimes L^{-1/2},\\
\vecop(AMC)&=(C^\top\otimes A)\vecop(M),\\
P_{\rm all}&=\mathcal Q^\top\operatorname{blockdiag}(\{P(X_j)\}_j)\mathcal Q,\quad\mathcal Q^\top\mathcal Q=I,\\
d_X(X,X')^2&=d_{\BW}(L,L')^2+d_{\BW}(R,R')^2,\\
\|X-X'\|_{\rm pair}^2&=\|L-L'\|_F^2+\|R-R'\|_F^2.
\end{align}
\begin{align}
G&=B^{-1}\sum_{i=1}^BH_i,\quad H_i\overset{\rm iid}\sim\mathcal P,\quad\bar H=\E H_i,\\
\E[GG^\top]&=B^{-2}\sum_{i,j}\E[H_iH_j^\top]\\
&=B^{-1}\E[HH^\top]+(1-B^{-1})\bar H\bar H^\top\\
&=\bar H\bar H^\top+B^{-1}\E[(H-\bar H)(H-\bar H)^\top],\\
X_t^\star&\in\argmin_X\tfrac12\E[d_X(X,S_t)^2\mid\mathcal F_{t-1}].
\label{eq:frechettarget}
\end{align}
''' + s[first:]
insertions={r'\subsection{Local stochastic state tracking with correct RMS propagation}':r'''
\begin{align}
V&=HA+AH,\quad H=H^\top,\quad Z=A^{1/2},\\
\Log_A B&=(T(A,B)-I)A+A(T(A,B)-I),\\
h_B(A)&=\tfrac12d_{\BW}(A,B)^2,\quad\grad h_B(A)=-\Log_A B.
\end{align}
''',r'\subsection{Lipschitz continuity of the normalized direction}':r'''
\begin{equation}
p_-I\preceq P,P'\preceq p_+I,\quad0<p_-\le p_+,\quad0<s_-\le s_+<\infty,\quad\varepsilon\ge0.
\end{equation}
''',r'\label{app:budget}':r'''
\begin{align}
\Sigma_t,\ s_t^\star&\in\mathcal F_{t-1},\quad a^2I\preceq\Sigma_t\preceq b^2I,\quad s_t^\star\in[s_-,s_+],\\
\bar B_t&=s_t^\star\Sigma_t^{-1/2},\quad B_t=s_t\widetilde P_t,\\
(L_t^\dagger,R_t^\dagger)&\in\argmin_{(L,R)\in\mathcal C_{a,b}}\|\Sigma_t-R\otimes L\|_\op,\\
\mathcal C_{a,b}&\subset\mathcal K_m(a,b)\times\mathcal K_n(a,b)\quad\text{compact},\\
b_-&=s_-/b,\quad b_+=s_+/a.
\end{align}
''',r'\subsection{Explicit Lipschitz constants for the Bures state map}':r'''
\begin{align}
\|G\|_F&\le G_0,\quad\varepsilon_{\rm stat}I\preceq S_L,S_R
\preceq(G_0^2/\min(m,n)+\varepsilon_{\rm stat})I,\\
a_0&=\min(a,\varepsilon_{\rm stat}),\quad
b_0=\max(b,G_0^2/\min(m,n)+\varepsilon_{\rm stat}).
\end{align}
''',r'\begin{theorem}[Full-state stability and expected generalization]':r'''
\begin{align}
\beta&:=\beta_1,\quad \rho_t=\mathbf1\{t=1\text{ or }t\equiv0\pmod{\tau_{\rm root}}\},\\
c_P&=(\sqrt2a^2)^{-1},\quad\gamma_t=(1-\beta^t)^{-1},\\
d_m&=2(1-\beta)G_0,\quad d_X=4q_SG_0^2c_d,\quad d_K=\rho_tc_Pd_X,\\
p_-&=1/b,\quad p_+=1/a,\quad L_m=s_+p_++1+p_+/p_-,\\
L_K&=(s_++b)G_0,\quad d_\theta=|1-\eta_t\lambda_{\rm wd}|.
\end{align}
'''}
for key,value in insertions.items():
    if key.startswith(r'\begin{theorem}'):
        assert key in s;s=s.replace(key,value+key,1)
    elif key in s:s=s.replace(key,key+'\n'+value,1)
# The normalized-direction subsection has a historical title; insert at the labeled map instead.
if r'\varepsilon\ge0.' not in s:
    anchor=r'\label{eq:normalizedmap}'
    end=s.index(r'\end{equation}',s.index(anchor))+len(r'\end{equation}')
    s=s[:end]+insertions[r'\subsection{Lipschitz continuity of the normalized direction}']+s[end:]
assumptions={
'ass:tracking':r'''\begin{assumption}[Conditional local tracking conditions]
\label{ass:tracking}
\begin{align}
F_t(X)&=\tfrac12\E_t d_X(X,S_{t+1})^2,\quad X_t^\star\in\argmin_{X\in\mathcal U}F_t(X),\\
\mu,L_B,\sigma_{B,t}&\text{ deterministic},\quad\mu>0,\quad
X_t,X_t^\star,\{\Exp_{X_t}(-u\grad h_{S_{t+1}}):0\le u\le\alpha\}\subset\mathcal U,\\
\E_t\sup_{X\in K}\|\grad h_{S_{t+1}}(X)\|_X&<\infty\quad(K\Subset\mathcal U),\quad
\E_t\grad h_{S_{t+1}}(X)=\grad F_t(X),\\
\langle\grad F_t(X),-\Log_X X_t^\star\rangle_X&\ge\mu d_X(X,X_t^\star)^2,\\
\|\grad F_t(X)\|_X&\le L_Bd_X(X,X_t^\star),\\
\E_t\|\grad h_{S_{t+1}}(X)-\grad F_t(X)\|_X^2&\le\sigma_{B,t}^2\quad(X\in\mathcal U).
\end{align}
\end{assumption}''',
'ass:opt':r'''\begin{assumption}[Smooth stochastic optimization]
\label{ass:opt}
\begin{align}
\mathcal V&\subset\R^p\text{ convex},\quad\{\theta_t,\theta_{t+1},z_t\}_t\subset\mathcal V,\\
\|\nabla f(x)-\nabla f(y)\|&\le L_f\|x-y\|\quad(x,y\in\mathcal V),\quad f\ge f_\star,\\
\bar B_t&\in\mathcal F_{t-1},\quad b_-I\preceq\bar B_t\preceq b_+I,\quad0<b_-\le b_+<\infty,\\
\E\|g_t\|^2+\E\|r_t\|^2&<\infty,\quad\eqref{eq:oracle}.
\end{align}
\end{assumption}''',
'ass:stab':r'''\begin{assumption}[Stability setting]
\label{ass:stab}
\begin{align}
S&\sim\mathcal D^N,\quad S\sim S'\Longleftrightarrow|\{i:Z_i\ne Z_i'\}|=1,\\
(\theta_1,m_0,X_0,K_0)&=(\theta_1',m_0',X_0',K_0')\perp(S,S'),\\
I_t&\overset{\rm iid}\sim\operatorname{Unif}\{1,\ldots,N\},\quad I_t'=I_t,\quad\omega_t'=\omega_t,\\
\theta_t,\theta_t'&\in\mathcal V\subset\R^p\text{ convex},\\
\|g(\theta,z)\|&\le G_0,\quad\|g(\theta,z)-g(\theta',z)\|\le L_g\|\theta-\theta'\|,\\
|\ell(\theta,z)-\ell(\theta',z)|&\le L_\ell\|\theta-\theta'\|\quad(\theta,\theta'\in\mathcal V),\\
\rho_t&=\rho_t'\text{ deterministic},\quad aI\preceq L_t,R_t,L_t',R_t'\preceq bI,\\
\mathbf1\{g_{t,j}\text{ present}\}&=\mathbf1\{g_{t,j}'\text{ present}\}=1,\\
N_{\rm reset}&=N_{\rm decomposition\ failure}=0.
\end{align}
\end{assumption}'''}
for label,replacement in assumptions.items():
    pattern=r'\\begin\{assumption\}[^\n]*\n\\label\{'+re.escape(label)+r'\}.*?\\end\{assumption\}'
    s,n=re.subn(pattern,lambda m:replacement,s,flags=re.S);assert n==1,label
# Protect the complete theorem/proof statements and all displayed mathematics.
protected=[]
pat=r'\\begin\{(proof|theorem|lemma|proposition|corollary|assumption|definition|align\*?|equation\*?|gather\*?)\}.*?\\end\{\1\}'
def protect(m):protected.append(m.group());return '\n@@BLOCK'+str(len(protected)-1)+'@@\n'
t=re.sub(pat,protect,s,flags=re.S)
kept=[]
for line in t.splitlines():
    z=line.strip()
    if not z or z.startswith(('@@BLOCK',r'\section',r'\subsection',r'\label','%')):kept.append(line)
t='\n'.join(kept)
t=re.sub(r'@@BLOCK(\d+)@@',lambda m:protected[int(m.group(1))],t)
# Explicit mathematical conditions previously carried in external prose.
t+=r'''
\subsection{Residual closure, predictor budgets, and admissible implications}
\begin{align}
B_t^{\rm mixed}&=\operatorname{blockdiag}(B_t^{\rm matrix},B_t^a),\quad
\frac{I}{G_0+\varepsilon_a}\preceq B_t^a\preceq\frac{I}{\varepsilon_a},\\
r_t^{\rm mixed}&=D_t^{\rm implemented}-\bar B_t^{\rm mixed}g_t,\\
r_t^{\rm decay}&=r_t+\lambda_{\rm wd}\theta_t,\quad
\E\|r_t^{\rm decay}\|^2\le2\E\|r_t\|^2+2\lambda_{\rm wd}^2\E\|\theta_t\|^2,\\
\bar B_t=B_t\in\mathcal F_{t-1},\quad\widehat m_t=g_t
&\Longrightarrow r_t=0,\\
\frac{\sum_t\eta_t\E\|r_t\|^2}{H_T}=O(T^{-1/2}),\quad\eta_t\asymp T^{-1/2}
&\Longrightarrow\frac{\sum_t\eta_t\E\|\nabla f(\theta_t)\|^2}{H_T}=O(T^{-1/2}),\\
\liminf_T\frac{\sum_t\eta_t\E\|r_t\|^2}{H_T}>0
&\not\Longrightarrow\lim_t\|\nabla f(\theta_t)\|=0,\\
\|\widehat m_t\|&\le G_0,\quad\|D_t\|\le s_+G_0/a,\quad0\le\eta_t\lambda_{\rm wd}\le1,\\
\|\theta_T\|&\le\|\theta_1\|+(s_+G_0/a)\sum_{t<T}\eta_t=:R_T,\\
R_T\perp S&\Longrightarrow\Theta_T\subseteq B(0,R_T)\text{ admissible in Theorem~\ref{thm:cover}},\\
\{R_1,\ldots,R_J\}\perp S
&\Longrightarrow\log(2/\delta)\mapsto\log(2J/\delta),\\
\E[gg^\top]&=\nabla f\nabla f^\top+\Sigma_\xi,\\
\|\Pi_{[a,b]}A-\Pi_{[a,b]}B\|_F\le\|A-B\|_F
&\not\Longrightarrow d_{\BW}(\Pi A,\Pi B)\le d_{\BW}(A,B),\\
aI\preceq L_t,R_t\preceq bI
&\not\Longrightarrow \sup_t\|W_t\|_\op<\infty,\\
B_t=B_t(g_t)&\not\Longrightarrow\E_{t-1}[B_tg_t]=\E_{t-1}[B_t]\nabla f(\theta_t).
\end{align}
\begin{proposition}[Fixed-preconditioner strongly convex stability]
\begin{gather}
B\succ0,\quad b_-I\preceq B\preceq b_+I,\quad
\mu_fI\preceq\nabla^2\ell(\theta,z)\preceq L_fI,\quad\eta\le1/(b_+L_f),\\
\mathcal T_z(\theta)=\theta-\eta B\nabla\ell(\theta,z),\quad
\|x\|_{B^{-1}}^2=x^\top B^{-1}x,\\
\|\mathcal T_z(\theta)-\mathcal T_z(\theta')\|_{B^{-1}}
\le(1-\eta b_-\mu_f)\|\theta-\theta'\|_{B^{-1}},\\
\epsilon_N\le\frac{2L_\ell b_+G_0}{Nb_-\mu_f}.
\end{gather}
\end{proposition}
\begin{proof}
\begin{align}
B^{-1/2}\mathrm D\mathcal T_z(\theta)B^{1/2}
&=I-\eta B^{1/2}\nabla^2\ell(\theta,z)B^{1/2},\\
0\preceq I-\eta B^{1/2}\nabla^2\ell B^{1/2}&\preceq(1-\eta b_-\mu_f)I,\\
\Delta_{t+1}&\le(1-\eta b_-\mu_f)\Delta_t+2\eta\sqrt{b_+}G_0/N,\\
\sup_t\Delta_t&\le\frac{2\sqrt{b_+}G_0}{Nb_-\mu_f},\quad
\|x\|\le\sqrt{b_+}\|x\|_{B^{-1}},\\
\epsilon_N&\le L_\ell\sqrt{b_+}\sup_t\Delta_t.
\end{align}
\end{proof}
'''
# Minimal side conditions attached directly to their conclusions.
t=t.replace('For uniform bounds this converges geometrically to the corresponding noise-and-residual floor. A nonvanishing residual cannot be dropped when stating linear convergence to an optimum.',r'For $\sigma_t^2\le\sigma^2$, $\E\|r_t\|^2\le\rho^2$ and $0\le1-\eta b_-\mu_f<1$, the following proof applies.')
t=re.sub(r'\n{4,}','\n\n\n',t)
p.write_text(t+'\n')
print('Formal definitions retained; exposition converted; scientific diagnostics moved to experimental appendix.')
