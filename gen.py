import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CHUNK_SIZE = 32
MICRO_SIZE = 4
MINI_SIZE = 16
NUM_CHUNKS_X = 16
NUM_CHUNKS_Y = 16
FONTSIZE = 64
THRESHOLD = 1 #def. 0.1 хз что это
ATOL = 1 #def. 1 тож хз что это азаза

def generate_chunk(size=CHUNK_SIZE):
    return np.random.rand(size, size)

def are_values_similar(values, threshold=THRESHOLD):
    printavs = np.max(values) - np.min(values) <= threshold
    print("AVS:", printavs)
    return printavs

def smooth_microchunk(chunk):
    center = chunk[1,1]
    if center == np.max(chunk) or center == np.min(chunk):
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                chunk[i,j] = 0.7 * center + 0.3 * chunk[i,j]
    return chunk

def smooth_minichunk(chunk):
    centers = chunk[7:9,7:9]
    max_val = np.max(centers)
    min_val = np.min(centers)
    if np.allclose(centers, max_val, atol=ATOL) or np.allclose(centers, min_val, atol=ATOL):
        target = np.mean(centers)
        chunk[:,:] = 0.7 * target + 0.3 * chunk
    return chunk

def apply_smoothing(chunk):
    size = chunk.shape[0]

    for i in range(0, size-2):
        for j in range(0, size-2):
            micro = chunk[i:i+3, j:j+3]
            micro = smooth_microchunk(micro)
            chunk[i:i+3, j:j+3] = micro

    for i in [0, 16]:
        for j in [0, 16]:
            mini = chunk[i:i+16, j:j+16]
            mini = smooth_minichunk(mini)
            chunk[i:i+16, j:j+16] = mini

    return chunk

def generate_multiple_chunks(num_x=NUM_CHUNKS_X, num_y=NUM_CHUNKS_Y):
    big_field = np.zeros((CHUNK_SIZE*num_y, CHUNK_SIZE*num_x))
    for y in range(num_y):
        for x in range(num_x):
            chunk = generate_chunk()
            chunk_smoothed = apply_smoothing(chunk)
            big_field[y*CHUNK_SIZE:(y+1)*CHUNK_SIZE, x*CHUNK_SIZE:(x+1)*CHUNK_SIZE] = chunk_smoothed
    return big_field

def compute_minichunk_averages(big_field, num_x=NUM_CHUNKS_X, num_y=NUM_CHUNKS_Y):
    # Размеры миничанков в большом поле
    mini_averages = np.zeros((num_y*2, num_x*2))  # 2 миничанка по 16 в одном чанке 32
    for cy in range(num_y):
        for cx in range(num_x):
            base_y = cy * CHUNK_SIZE
            base_x = cx * CHUNK_SIZE
            # В чанке 32x32 4 миничанка 16x16:
            for my in range(2):
                for mx in range(2):
                    mini = big_field[base_y + my*MINI_SIZE: base_y + (my+1)*MINI_SIZE,
                                     base_x + mx*MINI_SIZE: base_x + (mx+1)*MINI_SIZE]
                    mini_averages[cy*2 + my, cx*2 + mx] = np.mean(mini)
    return mini_averages

def plot_chunks_and_averages(big_field, num_x=NUM_CHUNKS_X, num_y=NUM_CHUNKS_Y):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(num_x*8, num_y*5))

    # Первый график — ландшафт с рамками
    im1 = ax1.imshow(big_field, cmap='terrain')

    for y in range(num_y):
        for x in range(num_x):
            # Рамка чанка
            rect = patches.Rectangle(
                (x*CHUNK_SIZE - 0.5, y*CHUNK_SIZE - 0.5),
                CHUNK_SIZE, CHUNK_SIZE,
                linewidth=2, edgecolor='black', facecolor='none'
            )
            ax1.add_patch(rect)
            
            # Рамки миничанков 16x16
            for mini_y in [0, 16]:
                for mini_x in [0, 16]:
                    rect_mini = patches.Rectangle(
                        (x*CHUNK_SIZE + mini_x - 0.5, y*CHUNK_SIZE + mini_y - 0.5),
                        MINI_SIZE, MINI_SIZE,
                        linewidth=1.5, edgecolor='cyan', facecolor='none', alpha=0.6
                    )
                    ax1.add_patch(rect_mini)
            #микрочанке
            for micro_y in range(0, CHUNK_SIZE - MICRO_SIZE + 1, MICRO_SIZE):
                for micro_x in range(0, CHUNK_SIZE - MICRO_SIZE + 1, MICRO_SIZE):
                    rect_micro = patches.Rectangle(
                        (x*CHUNK_SIZE + micro_x - 0.5, y*CHUNK_SIZE + micro_y - 0.5),
                        MICRO_SIZE, MICRO_SIZE,
                        linewidth=0.5, edgecolor='purple', facecolor='none', alpha=0.3
                    )
                    ax1.add_patch(rect_micro)

    ax1.set_title(f'{num_x}x{num_y} цвета дискатецка',fontsize= FONTSIZE)
    cb=fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Height')
    cb.ax.tick_params(labelsize=FONTSIZE/1.5)

    # Второй график — карта средних значений миничанков
    mini_avgs = compute_minichunk_averages(big_field, num_x, num_y)
    im2 = ax2.imshow(mini_avgs, cmap='terrain', interpolation='none')
    ax2.set_title('среднее значения миничанков (границы синее на соседнем графике)', fontsize= FONTSIZE)
    cb2=fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Avg Height')
    cb2.ax.tick_params(labelsize=FONTSIZE/1.5)

    plt.tight_layout()
    plt.savefig("allpixels.png", dpi=100)

big_field = generate_multiple_chunks()
plot_chunks_and_averages(big_field)
