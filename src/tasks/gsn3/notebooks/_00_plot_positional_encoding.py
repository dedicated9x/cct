import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('TkAgg')  # Or 'Agg' for non-interactive plots

from src.tasks.gsn3.arch import get_positional_encoding


tokens = 64
dimensions = 128

pos_encoding = get_positional_encoding(tokens, dimensions)

plt.figure(figsize=(12,8))
plt.pcolormesh(pos_encoding.numpy(), cmap='viridis')
# plt.pcolormesh(pos_encoding[0], cmap='viridis')
plt.xlabel('Embedding Dimensions')
plt.xlim((0, dimensions))
plt.ylim((tokens,0))
plt.ylabel('Token Position')
plt.colorbar()
plt.show()