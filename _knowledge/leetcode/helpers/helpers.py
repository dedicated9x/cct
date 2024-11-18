def int_to_32bit_binary(n):
    # For negative numbers, use two's complement by masking with 0xFFFFFFFF
    return format(n & 0xFFFFFFFF, '032b')