import torch
import numpy as np

Q = torch.load("/home/admin2/Documents/repos/cct/.EXCLUDED/outputs/qkv/Q.pt").detach().numpy()
K = torch.load("/home/admin2/Documents/repos/cct/.EXCLUDED/outputs/qkv/K.pt").detach().numpy()
V = torch.load("/home/admin2/Documents/repos/cct/.EXCLUDED/outputs/qkv/V.pt").detach().numpy()

single_q = Q[17, 4, :]
list_64_ks = K[:, 4, :]

list_dot_products = []
for k in list_64_ks:
    dot_product = (single_q * k).sum()
    list_dot_products.append(dot_product)

list_dot_products = torch.Tensor(list_dot_products)
print(list_dot_products)

QKt_1 = np.zeros((7, 64, 64))
for b in range(7):
    for i in range(64):
        for j in range(64):
            for k in range(16):
                QKt_1[b, i, j] += Q[i, b, k] * K[j, b, k]

QKt_2 = np.einsum('ibk,jbk->bij', Q, K)


assert torch.isclose(list_dot_products, torch.Tensor(QKt_2[4, 17, :])).all().item()

QKt = QKt_2