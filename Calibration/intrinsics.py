import numpy as np
#### 内参

# 这个是产生论文里的v矩阵
def v(p, q, H):
    return np.array([
        H[0, p] * H[0, q],
        H[0, p] * H[1, q] + H[1, p] * H[0, q],
        H[1, p] * H[1, q],
        H[2, p] * H[0, q] + H[0, p] * H[2, q],
        H[2, p] * H[1, q] + H[1, p] * H[2, q],
        H[2, p] * H[2, q]
    ])

def get_camera_intrinsics(homographies):

    h_count = len(homographies)

    # vec = []

    V=np.zeros((2*h_count,6),np.float64)
    # for i in range(h_count):
    #     # curr = np.reshape(homographies[i][0], (3, 3))
    #     curr = homographies[i]
    #
    #     vec.append(v(0, 1, curr))
    #     vec.append(v(0, 0, curr) - v(1, 1, curr))
    #
    # vec = np.array(vec)
    #
    # u,s,vh=np.linalg.svd(vec)
    # b=vh[np.argmin(s)]
    # b = np.linalg.lstsq(
    #     vec,
    #     np.zeros(h_count * 2),
    # )[-1]

    for i in range(h_count):
        H = homographies[i]
        V[2 * i] = v(p=0, q=1, H=H)
        V[2 * i + 1] = np.subtract(v(p=0, q=0, H=H), v(p=1, q=1, H=H))
    u, s, vh = np.linalg.svd(V)
    b = vh[np.argmin(s)]
    w = b[0] * b[2] * b[5] - b[1]**2 * b[5] - b[0] * b[4]**2 + 2 * b[1] * b[3] * b[4] - b[2] * b[3]**2
    d = b[0] * b[2] - b[1]**2
    # if (d < 0):
    #     d = 0.01
    # d = -d
    # alpha = np.sqrt(w / (d * b[0]))
    # beta = np.sqrt(w / d**2 * b[0])
    # gamma = np.sqrt(w / (d**2 * b[0])) * b[1]
    # uc = (b[1] * b[4] - b[2] * b[3]) / d
    # vc = (b[1] * b[3] - b[0] * b[4]) / d
    vc = (b[1] * b[3] - b[0] * b[4]) / d
    l=b[5]-(b[3]**2+vc*(b[1]*b[2]-b[0]*b[4]))/b[0]
    alpha = np.sqrt(l / ( b[0]))
    beta = np.sqrt(((l*b[0])/(b[0]*b[2]-b[1]**2)))
    gamma = -1*((b[1])*(alpha**2)*(beta/l))
    uc = (gamma*vc/beta)-(b[3]*(alpha**2)/l)

    return np.array([
        [alpha, gamma, uc],
        [0,     beta,  vc],
        [0,     0,      1]
    ]),b


homo2=[]
for i in homo:
    homo2.append(i[0])
homo2=np.array(homo2)

int2,b=get_camera_intrinsics(homo2)

homo=np.array(h_all)
int,b=get_camera_intrinsics(homo)
