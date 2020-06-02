def overlap_ratio(x_start, x_end, y_start, y_end):
    x_len = x_end - x_start
    y_len = y_end - y_start

    overlap_start = max(x_start, y_start)
    overlap_end = min(x_end, y_end)
    return max(overlap_end - overlap_start, 0.) / max(x_len, y_len)
