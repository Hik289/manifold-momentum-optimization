"""Replace expository proofs by explicit derivations; preserve the prior source in archive."""
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'archive/manuscript_before_flow_expansion/theory_proofs.tex'
proofs=[]
def add(s):proofs.append('\n'+s.strip()+'\n')
add(r'''\begin{align}
M&=B^{1/2}A^{1/2}=U\Lambda V^\top,\qquad O_\star=UV^\top,\\
\min_{O\in O(d)}\|A^{1/2}-B^{1/2}O\|_F^2
&=\tr A+\tr B-2\max_{O\in O(d)}\tr(O^\top M)\\
&=\tr A+\tr B-2\tr(M^\top M)^{1/2},\\
V_A&=(A^{1/2}BA^{1/2})^{1/2},\quad T=A^{-1/2}V_AA^{-1/2}\succ0,\\
TAT&=A^{-1/2}V_A^2A^{-1/2}=B,\\
H_s&=(1-s)I+sT\succ0,\quad Z_s=H_sA^{1/2},\\
Z_s^\top Z_t&=A^{1/2}H_sH_tA^{1/2}\succ0,\\
d_{\BW}(Z_sZ_s^\top,Z_tZ_t^\top)
&=\|Z_s-Z_t\|_F=|s-t|\|(T-I)A^{1/2}\|_F.
\end{align}
\begin{align}
ZZ^\top=A,\ YY^\top=B,\ WW^\top=C:\qquad
d(A,C)&\le\|Z-W\|_F\le\|Z-Y\|_F+\|Y-W\|_F\\
&=d(A,B)+d(B,C)
\end{align}
with $Z,Y$ and $Y,W$ optimally aligned.''')
add(r'''\begin{align}
D&=d_{\BW}(A,B),\quad x=d_{\BW}(C,A),\quad y=d_{\BW}(C,B),\quad x+y\ge D,\\
(1-\alpha)x^2+\alpha y^2
&=\alpha(1-\alpha)(x+y)^2+[(1-\alpha)x-\alpha y]^2\\
&\ge\alpha(1-\alpha)D^2,\\
C=\Gamma_\alpha(A,B)&\Longrightarrow(x,y)=(\alpha D,(1-\alpha)D),\\
\det(B^{1/2}A^{1/2})\ne0&\Longrightarrow O_\star=\operatorname{polar}(B^{1/2}A^{1/2})\text{ unique},\\
T&=Q\diag(\sqrt{b_i/a_i})Q^\top,\\
\lambda_i(\Gamma_\alpha)&=[(1-\alpha)\sqrt{a_i}+\alpha\sqrt{b_i}]^2,\\
(1-\alpha)a_i+\alpha b_i-\lambda_i(\Gamma_\alpha)
&=\alpha(1-\alpha)(\sqrt{a_i}-\sqrt{b_i})^2.
\end{align}''')
add(r'''\begin{align}
f(QAQ^\top)&=Qf(A)Q^\top,\quad f(x)\in\{\sqrt x,x^{-1/2}\},\\
T(QAQ^\top,QBQ^\top)&=QT(A,B)Q^\top,\\
\Gamma_\alpha(QAQ^\top,QBQ^\top)&=Q\Gamma_\alpha(A,B)Q^\top,\\
P(cL,c^{-1}R)&=(c^{1/2}R^{-1/2})\otimes(c^{-1/2}L^{-1/2})=P(L,R),\\
C&=\Pi_{[a,b]}A,\quad D=\Pi_{[a,b]}B,\\
\langle A-C,D-C\rangle_F&\le0,\quad\langle B-D,C-D\rangle_F\le0,\\
\|C-D\|_F^2&\le\langle C-D,A-B\rangle_F\le\|C-D\|_F\|A-B\|_F.
\end{align}''')
add(r'''\begin{align}
Z_u&=(1-u)Z_0+uZ_1,\quad Z_u^\top Z_v\succ0,\\
(1-\alpha)Z_u+\alpha Z_v&=Z_{(1-\alpha)u+\alpha v},\\
\Gamma_\alpha(\Gamma_u(A,B),\Gamma_v(A,B))
&=\Gamma_{(1-\alpha)u+\alpha v}(A,B),\\
u_t&=(1-\alpha)u_{t-1}+\alpha\tau_t,\\
d_{\BW}(C_t,C_t^\star)&=|u_t-\tau_t|\|Z_1-Z_0\|_F.
\end{align}''')
add(r'''\begin{align}
\beta&=1-\alpha,\quad e_t=u_t-y^\star,\quad\xi_t=Y_t-y^\star,\\
e_t&=\beta e_{t-1}+\alpha\xi_t,\quad \E[\xi_t\mid e_{t-1}]=0,\\
\E\|e_t\|^2&=\beta^2\E\|e_{t-1}\|^2+\alpha^2v\\
&=\beta^{2t}\|e_0\|^2+\alpha^2v\sum_{j=0}^{t-1}\beta^{2j}\\
&=\beta^{2t}\|e_0\|^2+\frac{\alpha(1-\beta^{2t})}{2-\alpha}v,\\
\argmin_{u>0}\E\|u-Y\|^2&=\E Y=y^\star,\\
\E Y_i^2-(\E Y_i)^2&=\Var(Y_i).
\end{align}''')
add(r'''\begin{align}
P=cI,\ m\ne0&\Longrightarrow \|Pm\|=c\|m\|,\\
\frac{\|m\|}{\|Pm\|}Pm&=\frac1c\,cm=m,\\
\operatorname{clip}\left(\frac{\|m\|}{\|Pm\|},s_-,s_+\right)Pm
&=\operatorname{clip}(c^{-1},s_-,s_+)cm.
\end{align}''')
add(r'''\begin{align}
\sqrt{c_t^B}&=\beta\sqrt{c_{t-1}^B}+\alpha\sqrt s
=\beta^t\sqrt{c_0}+(1-\beta^t)\sqrt s,\\
c_t^E&=\beta c_{t-1}^E+\alpha s=\beta^tc_0+(1-\beta^t)s,\\
P_t&=(c_tI)^{-1/2}\otimes(c_tI)^{-1/2}=c_t^{-1}I,\\
U_t&=\max\{c_t,s_-\}\,c_t^{-1}m_t,\\
c_t^B=s_-&\Longleftrightarrow\beta^t=\frac{\sqrt{s_-}-\sqrt s}{\sqrt{c_0}-\sqrt s},\\
c_t^E=s_-&\Longleftrightarrow\beta^t=\frac{s_--s}{c_0-s},\\
\lim_{s\downarrow0}\frac{t_B}{t_E}&=\frac12,\qquad
\lim_{t\to\infty}\frac{\|U_t\|}{\|m_t\|}=\frac{s_-}{s}.
\end{align}''')
add(r'''\begin{align}
Z&=A^{1/2},\quad Z_B=T(A,B)Z,\quad Z_+=(I-\alpha H)Z,\\
d_{\BW}(\Exp_A(-\alpha V),B)^2
&\le\|Z_+-Z_B\|_F^2\\
&=\|[I-T(A,B)]Z-\alpha HZ\|_F^2\\
&=d_{\BW}(A,B)^2-2\alpha\langle HZ,[I-T(A,B)]Z\rangle_F+\alpha^2\|HZ\|_F^2\\
&=d_{\BW}(A,B)^2-2\alpha\langle V,-\Log_A B\rangle_A+\alpha^2\|V\|_A^2,\\
V=-\Log_A S&\Longrightarrow I-\alpha H=(1-\alpha)I+\alpha T(A,S)\succ0.
\end{align}''')
add(r'''\begin{align}
V_t&=\grad h_{S_{t+1}}(X_t),\quad\E_tV_t=\grad F_t(X_t),\\
\E_t\|V_t\|_{X_t}^2
&=\|\grad F_t(X_t)\|_{X_t}^2+\E_t\|V_t-\grad F_t(X_t)\|_{X_t}^2\\
&\le L_B^2d_X(X_t,X^\star)^2+\sigma_{B,t}^2,\\
\E_td_X(X_{t+1},X^\star)^2
&\le d_X(X_t,X^\star)^2-2\alpha\langle\grad F_t(X_t),-\Log_{X_t}X^\star\rangle
+\alpha^2\E_t\|V_t\|^2\\
&\le qd_X(X_t,X^\star)^2+\alpha^2\sigma_{B,t}^2,\\
e_t&\le q^te_0+\alpha^2\sum_{j<t}q^{t-1-j}\sigma_{B,j}^2,\\
\limsup_te_t&\le\frac{\alpha^2\sigma_B^2}{1-q}
=\frac{\alpha\sigma_B^2}{2\mu-\alpha L_B^2}.
\end{align}''')
add(r'''\begin{align}
r_{t+1}&\le F_t(r_t)+\delta_t+c_t,\quad F_t(u)=\sqrt{qu^2+\alpha^2\sigma_{B,t}^2},\\
0\le F_t'(u)&=\frac{qu}{\sqrt{qu^2+\alpha^2\sigma_{B,t}^2}}\le\sqrt q,\\
v_0&=r_0,\quad v_{t+1}=F_t(v_t),\\
v_t^2&=q^tr_0^2+\alpha^2\sum_{j<t}q^{t-1-j}\sigma_{B,j}^2,\\
(r_{t+1}-v_{t+1})_+&\le\sqrt q(r_t-v_t)_++\delta_t+c_t,\\
r_t&\le v_t+\sum_{j<t}q^{(t-1-j)/2}(\delta_j+c_j),\\
\limsup_tr_t&\le\frac{\alpha\sigma_B}{\sqrt{1-q}}+\frac{\delta+c}{1-\sqrt q},\\
\Phi(\alpha)&=\frac{\alpha\sigma_B^2}{\mu}+\frac{\delta^2}{\mu^2\alpha^2},\quad
\Phi'(\alpha)=\frac{\sigma_B^2}{\mu}-\frac{2\delta^2}{\mu^2\alpha^3},\\
\Phi'(\alpha)=0&\Longleftrightarrow\alpha^3=\frac{2\delta^2}{\mu\sigma_B^2}.
\end{align}''')
add(r'''\begin{align}
ZZ^\top&=A,\ YY^\top=B,\quad\|Z-Y\|_F=d_{\BW}(A,B),\\
A-B&=(Z-Y)Z^\top+Y(Z-Y)^\top,\\
\|A-B\|_F&\le(\|Z\|_\op+\|Y\|_\op)\|Z-Y\|_F\le2\sqrt b\,d_{\BW}(A,B),\\
X&=A^{1/2}-B^{1/2},\quad A^{1/2}X+XB^{1/2}=A-B,\\
X&=\int_0^\infty e^{-uA^{1/2}}(A-B)e^{-uB^{1/2}}du,\\
d_{\BW}(A,B)&\le\|X\|_F\le\frac{\|A-B\|_F}{2\sqrt a},\\
A^{-1/2}-B^{-1/2}&=\frac1\pi\int_0^\infty u^{-1/2}(A+uI)^{-1}(B-A)(B+uI)^{-1}du,\\
\|A^{-1/2}-B^{-1/2}\|_\op
&\le\frac{\|A-B\|_\op}{\pi}\int_0^\infty\frac{u^{-1/2}}{(a+u)^2}du
=\frac{\|A-B\|_\op}{2a^{3/2}},\\
P(X)-P(X')&=(R^{-1/2}-R'^{-1/2})\otimes L^{-1/2}
+R'^{-1/2}\otimes(L^{-1/2}-L'^{-1/2}),\\
\|P(X)-P(X')\|_\op&\le\frac{\|R-R'\|_\op+\|L-L'\|_\op}{2a^2}.
\end{align}''')
add(r'''\begin{align}
E&=C-C^\star,\quad U=U_r(C),\quad V=U_r(C^\star)^\perp,\\
CU&=U\Lambda,\quad C^\star V=V\Lambda_\perp,\quad Z=V^\top U,\\
Z\Lambda-\Lambda_\perp Z&=V^\top EU,\\
\lambda_r(C)-\lambda_{r+1}(C^\star)&\ge\Delta-\|E\|_\op\ge\Delta/2,\\
\|Z\|_\op&\le2\|E\|_\op/\Delta,\quad\|Z\|_F\le\sqrt r\|Z\|_\op,\\
\|\sin\Theta\|_F&\le\frac{2\sqrt r}{\Delta}\|E\|_\op\le\frac{4\sqrt{rb}}{\Delta}d_{\BW}(C,C^\star),\\
0&\le r^{-1/2}\|\sin\Theta\|_F\le1.
\end{align}''')
add(r'''\begin{align}
\|P(X_t)-P(X_{r(t)})\|_\op
&\le\frac{\|L_t-L_{r(t)}\|_\op+\|R_t-R_{r(t)}\|_\op}{2a^2}\\
&\le\frac1{2a^2}\sum_{j=r(t)+1}^t(\|L_j-L_{j-1}\|_\op+\|R_j-R_{j-1}\|_\op),\\
\widetilde P&=(R_r^{-1/2}+E_R)\otimes(L_r^{-1/2}+E_L),\\
\|\widetilde P-P(X_r)\|_\op
&\le a^{-1/2}(\|E_R\|_\op+\|E_L\|_\op)+\|E_R\|_\op\|E_L\|_\op.
\end{align}''')
add(r'''\begin{align}
\mathcal C\text{ compact},\quad (L,R)\mapsto\|\Sigma-R\otimes L\|_\op\text{ continuous}
&\Longrightarrow\argmin_{(L,R)\in\mathcal C}\|\Sigma-R\otimes L\|_\op\ne\varnothing,\\
R\otimes L-R'\otimes L'&=(R-R')\otimes L'+R\otimes(L-L'),\\
\|R\otimes L-R'\otimes L'\|_\op
&\le b(\|R-R'\|_\op+\|L-L'\|_\op),\\
\|\Sigma-R\otimes L\|_\op
&\le\|\Sigma-R^\dagger\otimes L^\dagger\|_\op
+\|R^\dagger\otimes L^\dagger-R^\star\otimes L^\star\|_\op\\
&\quad+\|R^\star\otimes L^\star-R\otimes L\|_\op,\\
\lambda_{\min}(\Sigma),\lambda_{\min}(R\otimes L)\ge a^2
&\Longrightarrow\|\Sigma^{-1/2}-(R\otimes L)^{-1/2}\|_\op
\le\frac{\|\Sigma-R\otimes L\|_\op}{2a^3}.
\end{align}''')
add(r'''\begin{align}
r&=\|m\|,\ p=\|Pm\|,\ s=\clip(r/(p+\varepsilon),s_-,s_+),\quad p\ge p',\\
\|sPm-s'Pm'\|&\le s_+p_+\|m-m'\|+|s-s'|p',\\
|s-s'|p'&\le\frac{p'}{p+\varepsilon}|r-r'|
+\frac{r'p'}{(p+\varepsilon)(p'+\varepsilon)}|p-p'|\\
&\le|r-r'|+p_-^{-1}|p-p'|\\
&\le(1+p_+/p_-)\|m-m'\|,\\
\|U(P,m)-U(P,m')\|&\le(s_+p_++1+p_+/p_-)\|m-m'\|,\\
|s(P,m)-s(P',m)|\|P'm\|
&\le\frac{\|m\|\|P'm\|\|(P-P')m\|}{(\|Pm\|+\varepsilon)(\|P'm\|+\varepsilon)}\\
&\le p_-^{-1}\|(P-P')m\|,\\
\|U(P,m)-U(P',m)\|&\le(s_++p_-^{-1})\|m\|\|P-P'\|_\op,\\
\|U(P,m)\|&\le s_+p_+\|m\|\xrightarrow[m\to0]{}0.
\end{align}''')
add(r'''\begin{align}
h_t&=\nabla f(\theta_t),\quad\xi_t=g_t-h_t,\quad\E_{t-1}\xi_t=0,\\
f(\theta_{t+1})&\le f(\theta_t)-\eta_t\langle h_t,\bar B_tg_t+r_t\rangle
+L_f\eta_t^2(\|\bar B_tg_t\|^2+\|r_t\|^2),\\
\E_{t-1}\langle h_t,\bar B_tg_t\rangle&=\langle h_t,\bar B_th_t\rangle\ge b_-\|h_t\|^2,\\
|\langle h_t,r_t\rangle|&\le\tfrac14b_-\|h_t\|^2+b_-^{-1}\|r_t\|^2,\\
\E_{t-1}\|\bar B_tg_t\|^2&\le b_+^2(\|h_t\|^2+\sigma_t^2),\\
\frac{\eta_tb_-}2\E\|h_t\|^2
&\le\E[f(\theta_t)-f(\theta_{t+1})]+L_f\eta_t^2b_+^2\sigma_t^2
+(\eta_t/b_-+L_f\eta_t^2)\E\|r_t\|^2,\\
\frac2{b_-}(\eta_t/b_-+L_f\eta_t^2)&\le\frac{5\eta_t}{2b_-^2},\\
\|r_t\|^2&\le2\|B_t-\bar B_t\|_\op^2\|g_t\|^2
+2\|B_t\|_\op^2\|\widehat m_t-g_t\|^2.
\end{align}
\begin{equation}
\sum_{t=1}^T\E[f(\theta_t)-f(\theta_{t+1})]
=f(\theta_1)-\E f(\theta_{T+1})\le f(\theta_1)-f_\star.
\end{equation}''')
add(r'''\begin{align}
\Delta_t&=\E[f(\theta_t)-f_\star],\quad\|\nabla f(\theta)\|^2\ge2\mu_f(f(\theta)-f_\star),\\
\Delta_{t+1}&\le(1-\eta b_-\mu_f)\Delta_t
+L_f\eta^2b_+^2\sigma^2+(\eta/b_-+L_f\eta^2)\rho^2,\\
q_f&=1-\eta b_-\mu_f\in[0,1),\quad
c_f=L_f\eta^2b_+^2\sigma^2+(\eta/b_-+L_f\eta^2)\rho^2,\\
\Delta_{T+1}&\le q_f^T\Delta_1+c_f\sum_{j=0}^{T-1}q_f^j
\le q_f^T\Delta_1+\frac{c_f}{\eta b_-\mu_f}.
\end{align}''')
add(r'''\begin{align}
z_{t+1}&=\theta_t-(1+c_\beta)V_tm_t\\
&=\theta_t-\frac{V_t}{1-\beta}[\beta m_{t-1}+(1-\beta)g_t]\\
&=\theta_t-c_\beta V_tm_{t-1}-a_tB_tg_t\\
&=z_t-a_tB_tg_t+c_\beta(V_{t-1}-V_t)m_{t-1},\\
m_t&=(1-\beta)\sum_{j=1}^t\beta^{t-j}g_j,\quad
\|m_t\|\le G(1-\beta^t)\le G,\\
\|z_t-\theta_t\|&\le c_\beta\|V_{t-1}\|_\op\|m_{t-1}\|
\le c_\beta a_{t-1}b_+G.
\end{align}''')
add(r'''\begin{align}
\xi_t&=g_t-\nabla f(\theta_t),\quad\E_{t-1}\xi_t=0,\\
z_{t+1}&=z_t-a_t\{\bar B_t[\nabla f(z_t)+\xi_t]-R_t\},\\
z_t,\bar B_t&\in\mathcal F_{t-1},\quad\E_{t-1}\|\xi_t\|^2\le\sigma_t^2,\\
\E\|R_t\|^2&\le3G^2\E\|B_t-\bar B_t\|_\op^2
+3c_\beta^2G^2a_t^{-2}\E\|V_t-V_{t-1}\|_\op^2\\
&\quad+3b_+^2L_f^2\E\|\theta_t-z_t\|^2,\\
\|\nabla f(\theta_t)\|^2&\le2\|\nabla f(z_t)\|^2+2L_f^2\|\theta_t-z_t\|^2,\\
\frac1{A_T}\sum_ta_t\E\|\theta_t-z_t\|^2
&\le\frac{c_\beta^2b_+^2G^2}{A_T}\sum_{t\ge2}a_ta_{t-1}^2.
\end{align}
\begin{equation}
\text{Theorem~\ref{thm:descent}}\quad
(\theta_t,\eta_t,g_t,r_t)\leftarrow(z_t,a_t,\nabla f(z_t)+\xi_t,-R_t)
\quad\Longrightarrow\quad\eqref{eq:momentumbound}.
\end{equation}''')
add(r'''\begin{align}
r_t^2&\le4(J_{\rm init,t}+J_{\rm stat,t}+J_{\rm drift,t}+J_{\rm clip,t}),\\
\E\|B_t-\bar B_t\|_\op^2
&\le C\left[4\sum_{k\in\{\rm init,stat,drift,clip\}}J_{k,t}
+J_{\rm target,t}+J_{\rm kron,t}+J_{\rm cache,t}+J_{\rm norm,t}\right],\\
\E\|r_t^{\rm opt}\|^2
&\le2G^2\E\|B_t-\bar B_t\|_\op^2+2(s_+/a)^2J_{\rm mom,t},\\
\frac{\sum_t\eta_t\E\|\nabla f(\theta_t)\|^2}{H_T}
&\le\frac{2\Delta_f}{b_-H_T}+\frac{2L_fb_+^2\sum_t\eta_t^2\sigma_t^2}{b_-H_T}
+\frac{C_J}{H_T}\sum_t\eta_t\sum_kJ_{k,t}.
\end{align}''')
add(r'''\begin{align}
D&=A^{1/2}SA^{1/2},\quad a_0^2I\preceq D\preceq b_0^2I,\\
D-D'&=(A^{1/2}-A'^{1/2})SA^{1/2}
+A'^{1/2}(S-S')A^{1/2}+A'^{1/2}S'(A^{1/2}-A'^{1/2}),\\
\|D-D'\|_F&\le\frac{b_0^{3/2}}{\sqrt{a_0}}\|A-A'\|_F+b_0\|S-S'\|_F,\\
\|D^{1/2}-D'^{1/2}\|_F&\le\frac{\|D-D'\|_F}{2a_0},\\
\|T-T'\|_F&\le\frac{b_0}{a_0^2}\|A-A'\|_F+\frac{\|D-D'\|_F}{2a_0^2}\\
&\le C_{TA}\|A-A'\|_F+C_{TS}\|S-S'\|_F,\\
\|H\|_\op,\|H'\|_\op&\le h,\quad\|H-H'\|_F\le\alpha\|T-T'\|_F,\\
HAH-H'A'H'&=(H-H')AH+H'(A-A')H+H'A'(H-H'),\\
\|HAH-H'A'H'\|_F&\le h^2\|A-A'\|_F+2b_0h\alpha\|T-T'\|_F\\
&\le q_A\|A-A'\|_F+q_S\|S-S'\|_F,\\
\|\Pi_{[a,b]}C-\Pi_{[a,b]}D\|_F&\le\|C-D\|_F.
\end{align}''')
add(r'''\begin{align}
GG^\top-G'G'^\top&=(G-G')G^\top+G'(G-G')^\top,\\
\|GG^\top-G'G'^\top\|_F&\le2G_0\|G-G'\|_F,\\
\|S(G)-S(G')\|_{\rm pair}^2
&\le4G_0^2(n^{-2}+m^{-2})\|G-G'\|_F^2,\\
\Pr(I_t=i_\star)&=1/N,\\
\E\|g_t-g_t'\|&\le(1-N^{-1})L_g\E\|\theta_t-\theta_t'\|+2G_0/N\\
&\le L_g\E\|\theta_t-\theta_t'\|+2G_0/N,\\
\E\|S_t-S_t'\|_{\rm pair}&\le k_\theta\E\|\theta_t-\theta_t'\|+4G_0^2c_d/N.
\end{align}''')
add(r'''\begin{align}
\E\|g_t-g_t'\|&\le L_ge_1^\top v_t+2G_0/N,\\
\E\|m_t-m_t'\|&\le r_mv_t+d_m/N,\\
\E\|X_t-X_t'\|_{\rm pair}&\le r_Xv_t+d_X/N,\\
\E\|K_t-K_t'\|_\op&\le\rho_tc_P\E\|X_t-X_t'\|_{\rm pair}
+(1-\rho_t)e_4^\top v_t\\
&\le r_Kv_t+d_K/N,\\
\E\|\theta_{t+1}-\theta_{t+1}'\|
&\le d_\theta e_1^\top v_t+\eta_tL_m\gamma_t(r_mv_t+d_m/N)
+\eta_tL_K(r_Kv_t+d_K/N),\\
v_{T+1}&\preceq\frac1N\sum_{j=1}^T\left(\prod_{t=T}^{j+1}A_t\right)b_j,\\
\sup_{S\sim S',z}\E|\ell(\theta_{T+1},z)-\ell(\theta_{T+1}',z)|
&\le L_\ell e_1^\top v_{T+1},\\
\E[f(\theta_{T+1})-f_S(\theta_{T+1})]
&=\frac1N\sum_i\E[\ell(A(S^{(i)}),Z_i)-\ell(A(S),Z_i)].
\end{align}''')
add(r'''\begin{align}
|A_tv|&\preceq A_t|v|\preceq\|v\|_wA_tw
\preceq(1+c_t\eta_t)\|v\|_ww,\\
\left\|\prod_{t=T}^{j+1}A_t\right\|_{w\to w}
&\le\prod_{t=j+1}^T(1+c_t\eta_t)
\le\exp\left(\sum_{t=j+1}^Tc_t\eta_t\right),\\
\epsilon_T&\le\frac{L_\ell w_1}{N}\sum_{j=1}^TD_j\eta_j
\exp\left(\sum_{t=j+1}^Tc_t\eta_t\right),\\
A_tw\preceq\rho w,\ \|b_t\|_w\le D
&\Longrightarrow\epsilon_T\le\frac{L_\ell w_1D}{N}\sum_{k=0}^{T-1}\rho^k
\le\frac{L_\ell w_1D}{N(1-\rho)}.
\end{align}''')
add(r'''\begin{align}
r_{t,i}^a&=\left\|(\sqrt{\gamma_a}r_{t-1,i}^a,\sqrt{1-\gamma_a}g_{t,i}^a)\right\|_2,\\
|r_{t,i}^a-r_{t,i}^{a\prime}|
&\le\left[\gamma_a(r_{t-1,i}^a-r_{t-1,i}^{a\prime})^2
+(1-\gamma_a)(g_{t,i}^a-g_{t,i}^{a\prime})^2\right]^{1/2},\\
\|r_t^a-r_t^{a\prime}\|&\le\sqrt{\gamma_a}\|r_{t-1}^a-r_{t-1}^{a\prime}\|
+\sqrt{1-\gamma_a}\|g_t^a-g_t^{a\prime}\|,\\
D_{t,i}&=r_{t,i}^a/\sqrt{1-\gamma_a^t}+\varepsilon_a\ge\varepsilon_a,\\
u_t^a&=\frac{m_t^a}{(1-\beta_a^t)D_t},\quad\|\widehat m_t^a\|\le G_0,\\
\|u_t^a-u_t^{a\prime}\|
&\le\frac{\|m_t^a-m_t^{a\prime}\|}{\varepsilon_a(1-\beta_a^t)}
+\frac{G_0\|r_t^a-r_t^{a\prime}\|}{\varepsilon_a^2\sqrt{1-\gamma_a^t}},\\
\widetilde v_{t+1}&\preceq\widetilde A_t\widetilde v_t+\widetilde b_t/N,\\
\widetilde v_{T+1}&\preceq\frac1N\sum_{j=1}^T\left(\prod_{t=T}^{j+1}\widetilde A_t\right)\widetilde b_j.
\end{align}''')
add(r'''\begin{align}
\mathcal N_\varepsilon&\subset\Theta,\quad\|u-v\|>\varepsilon\quad(u\ne v\in\mathcal N_\varepsilon),\\
\bigsqcup_{u\in\mathcal N_\varepsilon}B(u,\varepsilon/2)&\subset B(0,R+\varepsilon/2),\\
|\mathcal N_\varepsilon|(\varepsilon/2)^p&\le(R+\varepsilon/2)^p,\quad
|\mathcal N_\varepsilon|\le(1+2R/\varepsilon)^p,\\
\Pr\left(\max_{u\in\mathcal N_\varepsilon}|f(u)-f_S(u)|>v\right)
&\le2|\mathcal N_\varepsilon|e^{-2Nv^2},\\
\forall\theta\in\Theta\quad\exists u\in\mathcal N_\varepsilon:\quad\|\theta-u\|&\le\varepsilon,\\
|f(\theta)-f_S(\theta)|&\le2L_\ell\varepsilon+|f(u)-f_S(u)|,\\
v^2&=\frac{p\log(1+2R/\varepsilon)+\log(2/\delta)}{2N}.
\end{align}''')
add(r'''\begin{align}
h_W(x)-h_{W'}(x)&=\sum_{l=1}^D\big[h_{(W_1',\ldots,W_{l-1}',W_l,\ldots,W_D)}(x)
-h_{(W_1',\ldots,W_l',W_{l+1},\ldots,W_D)}(x)\big],\\
\|h_W(x)-h_{W'}(x)\|
&\le B_x\sum_{l=1}^D\left(\prod_{j\ne l}s_j\right)\|W_l-W_l'\|_F\\
&\le B_x\left[\sum_{l=1}^D\prod_{j\ne l}s_j^2\right]^{1/2}
\left[\sum_{l=1}^D\|W_l-W_l'\|_F^2\right]^{1/2},\\
|\ell(h_W(x),y)-\ell(h_{W'}(x),y)|&\le L_o\|h_W(x)-h_{W'}(x)\|,\\
\|\theta\|^2&=\sum_l\|W_l\|_F^2\le\sum_lF_l^2=R^2.
\end{align}''')
add(r'''\begin{align}
B_t&\in\mathcal F_{t-1},\quad\E_{t-1}\xi_t=0,\quad\E_{t-1}\xi_t\xi_t^\top=\Sigma_{\xi,t},\\
\E_{t-1}\|B_t\xi_t\|^2&=\E_{t-1}\tr(B_t\xi_t\xi_t^\top B_t^\top)
=\tr(B_t\Sigma_{\xi,t}B_t^\top),\\
\Sigma_{\xi,t}&=Q\diag(\lambda_i)Q^\top,\quad
B_t=Q\diag((\lambda_i+\lambda)^{-1/2})Q^\top,\\
\tr(B_t\Sigma_{\xi,t}B_t^\top)&=\sum_i\frac{\lambda_i}{\lambda_i+\lambda}.
\end{align}''')
add(r'''\begin{align}
\E[\theta_t\xi_t^\top]&=0,\quad C_{t+1}=AC_tA^\top+\eta^2B\Sigma B,\\
C_{t+1}&=A^tC_1(A^\top)^t+\eta^2\sum_{k=0}^{t-1}A^kB\Sigma B(A^\top)^k,\\
\rho(A)<1&\Longrightarrow\lim_tA^tC_1(A^\top)^t=0,\\
C_i&=(1-\eta b_ih_i)^2C_i+\eta^2b_i^2\sigma_i^2,\\
C_i&=\frac{\eta^2b_i^2\sigma_i^2}{1-(1-\eta b_ih_i)^2}
=\frac{\eta b_i\sigma_i^2}{h_i(2-\eta b_ih_i)},\\
\E f(\theta_\infty)&=\tfrac12\tr(HC)=\frac12\sum_i\frac{\eta b_i\sigma_i^2}{2-\eta b_ih_i}.
\end{align}''')
assert len(proofs)==29
s=source.read_text();i=iter(proofs)
s=re.sub(r'\\begin\{proof\}.*?\\end\{proof\}',lambda m:'\\begin{proof}'+next(i)+'\\end{proof}',s,flags=re.S)
(ROOT/'paper/sections/theory_proofs.tex').write_text(s)
print('Replaced 29 proofs by explicit mathematical derivations.')
