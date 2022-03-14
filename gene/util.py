def split_into_batchs(items, batch_size):
    if len(items) <= batch_size:
        return [items]
    else:
        return [items[:batch_size]] + split_into_batchs(items[batch_size:], batch_size)


def flatten_2d_list(nested_list):
    return [item for items in nested_list for item in items]
