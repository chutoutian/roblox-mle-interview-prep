


def motion_search(reference, current,search_range):
    """
    Perform motion search between reference and current frames.

    Args:
        reference (numpy.ndarray): The reference frame.
        current (numpy.ndarray): The current frame.
    
    
    Returns:        motion_vectors (numpy.ndarray): The motion vectors for each block.
    """
    h = len(current)
    w = len(current[0])

    block_size = 16
    motion_vectors = [[0] * (w // block_size) for _ in range(h // block_size)]

    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            best_mv = (0, 0)
            min_error = float('inf')

            for mv_x in range(-search_range, search_range + 1):
                for mv_y in range(-search_range, search_range + 1):
                    ref_block_x = j + mv_x
                    ref_block_y = i + mv_y

                    if (0 <= ref_block_x < w - block_size) and (0 <= ref_block_y < h - block_size):
                        ref_block = reference[ref_block_y:ref_block_y + block_size, ref_block_x:ref_block_x + block_size]
                        curr_block = current[i:i + block_size, j:j + block_size]

                        error = sum((ref_block - curr_block) ** 2)

                        if error < min_error:
                            min_error = error
                            best_mv = (mv_x, mv_y)

            motion_vectors[i // block_size][j // block_size] = best_mv

    return motion_vectors
