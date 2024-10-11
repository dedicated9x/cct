import torch
import torch.nn.functional as F
import numpy as np

Q = torch.load("/home/admin2/Documents/repos/cct/.EXCLUDED/outputs/qkv/Q.pt")
K = torch.load("/home/admin2/Documents/repos/cct/.EXCLUDED/outputs/qkv/K.pt")
V = torch.load("/home/admin2/Documents/repos/cct/.EXCLUDED/outputs/qkv/V.pt")

single_q = Q[17, 4, :]
list_64_ks = K[:, 4, :]

list_dot_products = []
for k in list_64_ks:
    dot_product = (single_q * k).sum()
    list_dot_products.append(dot_product)

list_dot_products = torch.Tensor(list_dot_products)
print(list_dot_products)

# QKt_1 = np.zeros((7, 64, 64))
# for b in range(7):
#     for i in range(64):
#         for j in range(64):
#             for k in range(16):
#                 QKt_1[b, i, j] += Q[i, b, k] * K[j, b, k]

QKt_2 = torch.einsum('ibk,jbk->bij', Q, K)


assert torch.isclose(list_dot_products, torch.Tensor(QKt_2[4, 17, :])).all().item()

QKt = QKt_2
dk = 16
QKt_norm = F.softmax(QKt / np.sqrt(dk), dim=2)

weights_single_pos= QKt_norm[4, 17, :]
V_single_sequence = V[:, 4, :]
list_weighted_Vs = []
for w, v in zip(weights_single_pos, V_single_sequence):
    list_weighted_Vs.append(w * v)
attention_single_pos_v1 = torch.stack(list_weighted_Vs).sum(dim=0)
print(attention_single_pos_v1)
attention_single_pos_v2 = weights_single_pos.unsqueeze(0) @ V_single_sequence
print(attention_single_pos_v2)

attention_slow = torch.zeros((64, 7, 16))
for b in range(7):
    for i in range(64):
        attention_slow[i, b, :] = QKt_norm[b, i, :].unsqueeze(0) @ V[:, b, :]

print(attention_slow[17, 4])

attention = torch.einsum('bik,kbj->ibj', QKt_norm, V)
print(attention[17, 4])
