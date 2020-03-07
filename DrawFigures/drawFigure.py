import matplotlib.pyplot as plt
import matplotlib

# sudo apt install msttcorefonts -qq

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font',family='Times New Roman')

x = [50, 100, 150, 200, 250, 300]
# y1 = [0.102, 0.127, 0.153, 0.183, 0.209, 0.214]
# y2 = [0.178, 0.274, 0.365, 0.428, 0.472, 0.481]
# y3 = [0.115, 0.160, 0.208, 0.231, 0.282, 0.285]
# y4 = [0.158, 0.214, 0.267, 0.302, 0.331, 0.352]
# y5 = [0.139, 0.196, 0.244, 0.280, 0.308, 0.315]
# y6 = [0.165, 0.221, 0.321, 0.377, 0.414, 0.429]
# y7 = [0.149, 0.203, 0.276, 0.306, 0.341, 0.359]
# y8 = [0.130, 0.192, 0.227, 0.290, 0.327, 0.333]
# y9 = [0.201, 0.320, 0.390, 0.452, 0.489, 0.501]

# y1 = [0.059,0.079,0.105,0.116,0.120,0.122]
# y2 = [0.135,0.182,0.228,0.247,0.268,0.289]
# y3 = [0.060,0.073,0.116,0.127,0.157,0.165]
# y4 = [0.087,0.112,0.156,0.168,0.174,0.189]
# y5 = [0.073,0.108,0.130,0.136,0.158,0.166]
# y6 = [0.127,0.164,0.186,0.216,0.245,0.264]
# y7 = [0.079,0.106,0.148,0.155,0.162,0.169]
# y8 = [0.072,0.120,0.130,0.140,0.164,0.167]
# y9 = [0.148,0.195,0.246,0.271,0.281,0.302]

# y1 = [0.104,0.139,0.157,0.174,0.191,0.213]
# y2 = [0.147,0.234,0.282,0.323,0.355,0.367]
# y3 = [0.118,0.165,0.197,0.228,0.241,0.254]
# y4 = [0.139,0.183,0.212,0.249,0.276,0.288]
# y5 = [0.140,0.184,0.215,0.253,0.280,0.291]
# y6 = [0.151,0.238,0.279,0.301,0.328,0.353]
# y7 = [0.143,0.203,0.256,0.297,0.322,0.341]
# y8 = [0.133,0.195,0.240,0.280,0.297,0.308]
# y9 = [0.163,0.267,0.324,0.364,0.383,0.392]

y1 = [0.051,0.069,0.081,0.089,0.098,0.106]
y2 = [0.099,0.158,0.196,0.232,0.248,0.257]
y3 = [0.067,0.092,0.119,0.128,0.134,0.147]
y4 = [0.071,0.101,0.125,0.147,0.161,0.171]
y5 = [0.071,0.095,0.121,0.143,0.156,0.164]
y6 = [0.092,0.148,0.176,0.199,0.218,0.245]
y7 = [0.084,0.116,0.141,0.160,0.182,0.192]
y8 = [0.073,0.114,0.138,0.164,0.176,0.185]
y9 = [0.102,0.172,0.217,0.243,0.261,0.272]

plt.figure()
l1 = plt.plot(x, y1, color='black', linewidth=2.0, linestyle='-',   label = 'CF'    )
l2 = plt.plot(x, y2, color='red', linewidth=2.0, linestyle='-',     label = 'RCTR'  )
l3 = plt.plot(x, y3, color='yellow', linewidth=2.0, linestyle='-',  label = 'SLIM'  )
l4 = plt.plot(x, y4, color='blue', linewidth=2.0, linestyle='-',    label = 'SRMP'  )
l5 = plt.plot(x, y5, color='gray', linewidth=2.0, linestyle='-',    label = 'LSMC'  )
l6 = plt.plot(x, y6, color='green', linewidth=2.0, linestyle='-',   label = 'TPMF-CF'   )
l7 = plt.plot(x, y7, color='purple', linewidth=2.0, linestyle='-',  label = 'CRABSN'    )
l8 = plt.plot(x, y8, color='brown', linewidth=2.0, linestyle='-',   label = 'LSMF-PR-b' )
l9 = plt.plot(x, y9, color='pink', linewidth=2.0, linestyle='-',    label = 'LSMF-PR'   )

for i, j in zip(x, y1):
    plt.scatter(i, j, marker='>',color="black")
for i, j in zip(x, y2):
    plt.scatter(i, j, marker='o',color="red")
for i, j in zip(x, y3):
    plt.scatter(i, j, marker='s',color="yellow")
for i, j in zip(x, y4):
    plt.scatter(i, j, marker='*',color="blue")
for i, j in zip(x, y5):
    plt.scatter(i, j, marker='D',color="gray")
for i, j in zip(x, y6):
    plt.scatter(i, j, marker='+',color="green")
for i, j in zip(x, y7):
    plt.scatter(i, j, marker='x',color="purple")
for i, j in zip(x, y8):
    plt.scatter(i, j, marker='1',color="brown")
for i, j in zip(x, y9):
    plt.scatter(i, j, marker='^',color="pink")

plt.xlabel('N')
# plt.ylabel('NDGG@N')
plt.ylabel('Recall@N')

plt.legend(labels=['CF','RCTR','SLIM','SRMP','LSMC','TPMF-CF','CRABSN','LSMF-PR-b','LSMF-PR'],  loc='best', fontsize=8, framealpha=1, fancybox=False)

plt.grid()  # 生成网格
# plt.title("NDGG@N on Epinions")
# plt.title("Recall@N on Epinions")
# plt.title("NDGG@N on Flixster")
plt.title("Recall@N on Flixster")

# plt.show()
plt.savefig("/workspace/4.pdf")