import numpy as np
def forward(A,B,pi,T,N,o):
    alpha=np.zeros((T,N))
    for i in range(N):
        alpha[0][i]=pi[i]*B[i][o[0]]
    for t in range(T-1):
        for i in range(N):
            sum_temp = 0
            for j in range(N):
                sum_temp=sum_temp+alpha[t][j]*A[j][i]
            alpha[t+1][i]=sum_temp*B[i][o[t+1]]
    p_o_lamda=alpha[T-1].sum(axis=0)
    return p_o_lamda,alpha
def backward(A,B,pi,T,N,o):
    beta=np.ones((T,N))
    for t in range(T-1,0,-1):
        for i in range(N):
            beta[t-1][i]=0
            for j in range(N):
                beta[t-1][i]=beta[t-1][i]+A[i][j]*B[j][o[t]]*beta[t][j]
    p_o_lamda=0
    for i in range(N):
        p_o_lamda=p_o_lamda+pi[i]*B[i][o[0]]*beta[0][i]
    return p_o_lamda,beta
if __name__ == '__main__':
    A=[[0.5,0.1,0.4],[0.3,0.5,0.2],[0.2,0.2,0.6]]
    B=[[0.5,0.5],[0.4,0.6],[0.7,0.3]]
    pi=[0.2,0.3,0.5]
    T=8
    N=3
    # 红色0，白色1
    o=[0,1,0,0,1,0,1,1]
    forward_p,alpha=forward(A,B,pi,T,N,o)
    backward_p,beta=backward(A,B,pi,T,N,o)
    # 求P(i4=q3|O,λ)
    i=3-1
    t=4-1
    densum=0
    for j in range(N):
        densum=densum+alpha[t][j]*beta[t][j]
    gamma43=alpha[t][i]*beta[t][i]/densum
    print('P(i_4=q_3|0,λ) =',gamma43)