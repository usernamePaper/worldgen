import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CHUNK_SIZE = 128
MINI_SIZE = 32
MICRO_SIZE = int(MINI_SIZE / 2)
NUM_CHUNKS_X = 8
NUM_CHUNKS_Y = NUM_CHUNKS_X
FONTSIZE = 64
THRESHOLD = 0.1 #def. 0.1 хз что это
ATOL = 1 #def. 1 тож хз что это азаза

def generate_chunk(size=CHUNK_SIZE):
    return np.random.rand(size, size)

def are_values_similar(values, threshold=THRESHOLD):
    printavs = np.max(values) - np.min(values) <= threshold
    print("AVS:", printavs)
    return printavs

def smooth_microchunk(chunk):
    size = chunk.shape[0]
    center_x = size // 2
    center_y = size // 2
    center_value = chunk[center_y, center_x]

    if center_value == np.max(chunk) or center_value == np.min(chunk):
        for i in range(size):
            for j in range(size):
                if i == center_y and j == center_x:
                    continue
                chunk[i, j] = 0.7 * center_value + 0.3 * chunk[i, j] #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ВАЖНАЯ ШТУКА эта формула влияет на генерацию. можно менять
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

    # Микрочанки
    for i in range(0, size - MICRO_SIZE + 1, MICRO_SIZE):
        for j in range(0, size - MICRO_SIZE + 1, MICRO_SIZE):
            micro = chunk[i:i+MICRO_SIZE, j:j+MICRO_SIZE]
            micro = smooth_microchunk(micro.copy())  # .copy() чтобы не трогать исходный срез напрямую
            chunk[i:i+MICRO_SIZE, j:j+MICRO_SIZE] = micro

    # Миничанки
    for i in range(0, size - MINI_SIZE + 1, MINI_SIZE):
        for j in range(0, size - MINI_SIZE + 1, MINI_SIZE):

            mini = chunk[i:i+MINI_SIZE, j:j+MINI_SIZE]
            mini = smooth_minichunk(mini.copy())
            chunk[i:i+MINI_SIZE, j:j+MINI_SIZE] = mini

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
    total_height = big_field.shape[0]
    total_width = big_field.shape[1]

    num_minichunks_y = total_height // MINI_SIZE
    num_minichunks_x = total_width // MINI_SIZE

    mini_averages = np.zeros((num_minichunks_y, num_minichunks_x))

    for y in range(num_minichunks_y):
        for x in range(num_minichunks_x):
            mini = big_field[
                y * MINI_SIZE : (y + 1) * MINI_SIZE,
                x * MINI_SIZE : (x + 1) * MINI_SIZE
            ]
            mini_averages[y, x] = np.mean(mini)
    
    return mini_averages


def plot_chunks_and_averages(big_field, num_x=NUM_CHUNKS_X, num_y=NUM_CHUNKS_Y):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(num_x*8, num_y*5))

    # == Левая часть: основной ландшафт ==
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
            
            # Рамки миничанков
            for mini_y in range(0, CHUNK_SIZE, MINI_SIZE):
                for mini_x in range(0, CHUNK_SIZE, MINI_SIZE):
                    rect_mini = patches.Rectangle(
                        (x*CHUNK_SIZE + mini_x - 0.5, y*CHUNK_SIZE + mini_y - 0.5),
                        MINI_SIZE, MINI_SIZE,
                        linewidth=1.5, edgecolor='cyan', facecolor='none', alpha=0.6
                    )
                    ax1.add_patch(rect_mini)

            # Рамки микрочанков
            for micro_y in range(0, CHUNK_SIZE, MICRO_SIZE):
                for micro_x in range(0, CHUNK_SIZE, MICRO_SIZE):
                    rect_micro = patches.Rectangle(
                        (x*CHUNK_SIZE + micro_x - 0.5, y*CHUNK_SIZE + micro_y - 0.5),
                        MICRO_SIZE, MICRO_SIZE,
                        linewidth=0.5, edgecolor='purple', facecolor='none', alpha=0.3
                    )
                    ax1.add_patch(rect_micro)

    ax1.set_title(f'{num_x}x{num_y} цвета дискатецка', fontsize=FONTSIZE)
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Height')
    cb1.ax.tick_params(labelsize=FONTSIZE / 1.5)

    # == Правая часть: средние миничанки ==
    total_height = big_field.shape[0]
    total_width = big_field.shape[1]

    num_minichunks_y = total_height // MINI_SIZE
    num_minichunks_x = total_width // MINI_SIZE

    mini_averages = np.zeros((num_minichunks_y, num_minichunks_x))
    for y in range(num_minichunks_y):
        for x in range(num_minichunks_x):
            mini = big_field[
                y * MINI_SIZE : (y + 1) * MINI_SIZE,
                x * MINI_SIZE : (x + 1) * MINI_SIZE
            ]
            mini_averages[y, x] = np.mean(mini)

    im2 = ax2.imshow(mini_averages, cmap='terrain', interpolation='none')
    ax2.set_title('Средние значения миничанков', fontsize=FONTSIZE)
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Avg Height')
    cb2.ax.tick_params(labelsize=FONTSIZE / 1.5)

    plt.tight_layout()
    plt.savefig("allpixels.png", dpi=100)


big_field = generate_multiple_chunks()
plot_chunks_and_averages(big_field)
