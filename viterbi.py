import numpy as np
def viterbi(A,B,pi,T,N,o):
    i_aster=np.zeros(T)
    delta=np.zeros((T,N))
    Phi=np.zeros((T,N))
    for i in range(N):
        delta[0][i]=pi[i]*B[i][o[0]]
        Phi[0][i]=0
    for t in range(1,T):
        for i in range(N):
            a=[]
            for j in range(N):
                a.append(delta[t-1][j]*A[j][i])
            delta[t][i]=np.max(a)*B[i][o[t]]
            Phi[t][i]=np.argmax(a,axis=0)+1
    P=np.max(delta[T-1])
    i_aster[T-1]=np.argmax(delta[T-1],axis=0)+1
    for t in range(T-2,-1,-1):
        i_aster[t]=Phi[t+1][int(i_aster[t+1]-1)]
    return i_aster,P
if __name__ == '__main__':
    A = [[0.5, 0.1, 0.4], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]]
    B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
    pi = [0.2, 0.3, 0.5]
    T = 8
    N = 3
    # 红色0，白色1
    o = [0, 1, 0, 0, 1, 0, 1, 1]
    I, P = viterbi(A, B, pi, T,N,o)
    print('Best route :',I)
    print('Probability of the best route :',P)


