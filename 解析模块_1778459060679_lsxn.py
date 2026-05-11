# parser_module.py
"""
解析模块：任意文件 ↔ UTF-8 十六进制文本
实现无损转换，不依赖文件类型，可创建过程性文件。
"""

import os

def file_to_hex(input_path: str, output_hex_path: str = None, chunk_size: int = 1024 * 1024):
    """
    将任意文件转换为十六进制文本。
    参数:
        input_path: 原始文件路径
        output_hex_path: 输出的十六进制文本文件路径（可选，不指定则自动生成）
        chunk_size: 分块读取大小（字节），默认1MB
    返回:
        生成的十六进制文本文件路径
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"文件不存在: {input_path}")

    if output_hex_path is None:
        base, _ = os.path.splitext(input_path)
        output_hex_path = base + ".hex.txt"

    with open(input_path, "rb") as fin, open(output_hex_path, "w", encoding="utf-8") as fout:
        while True:
            chunk = fin.read(chunk_size)
            if not chunk:
                break
            fout.write(chunk.hex())
    print(f"✅ 文件已转换为十六进制文本: {output_hex_path}")
    return output_hex_path


def hex_to_file(hex_input: str, output_path: str, is_hex_file: bool = True, chunk_size: int = 1024 * 1024):
    """
    将十六进制文本还原为原始文件。
    参数:
        hex_input: 十六进制字符串 或 十六进制文本文件路径（当 is_hex_file=True）
        output_path: 还原后文件的输出路径
        is_hex_file: True 表示 hex_input 是文件路径，False 表示是字符串
    """
    if is_hex_file:
        if not os.path.exists(hex_input):
            raise FileNotFoundError(f"十六进制文件不存在: {hex_input}")
        with open(hex_input, "r", encoding="utf-8") as fin, open(output_path, "wb") as fout:
            while True:
                chunk = fin.read(chunk_size * 2)  # 两个十六进制字符代表一个字节，所以读取字符数要加倍
                if not chunk:
                    break
                fout.write(bytes.fromhex(chunk))
    else:
        # 直接处理十六进制字符串
        with open(output_path, "wb") as fout:
            fout.write(bytes.fromhex(hex_input))

    print(f"✅ 十六进制已还原为文件: {output_path}")
    return output_path


# ---------- 命令行测试 ----------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法:")
        print("  编码: python parser_module.py encode <输入文件> [输出十六进制文件]")
        print("  解码: python parser_module.py decode <十六进制文件/字符串> <输出文件>")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "encode":
        inp = sys.argv[2]
        out = sys.argv[3] if len(sys.argv) > 3 else None
        file_to_hex(inp, out)
    elif mode == "decode":
        inp = sys.argv[2]
        out = sys.argv[3] if len(sys.argv) > 3 else "restored.bin"
        # 自动判断是文件还是字符串
        if os.path.exists(inp):
            hex_to_file(inp, out, is_hex_file=True)
        else:
            hex_to_file(inp, out, is_hex_file=False)
    else:
        print("未知模式: 请使用 encode 或 decode")