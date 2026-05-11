# token_module.py
"""
词元模块：实现字符串 ↔ UTF-8 十六进制编码的双向转换。
"""

def str_to_hex(text: str) -> str:
    """
    将字符串按 UTF-8 编码转换为连续十六进制字符串（无分隔符）。
    屏幕打印转换结果，并返回十六进制串。
    """
    hex_str = text.encode("utf-8").hex()
    print(f"字符串 → 十六进制: {hex_str}")
    return hex_str

def hex_to_str(hex_str: str) -> str:
    """
    将连续十六进制字符串按 UTF-8 解码为原始字符串。
    屏幕打印解码结果，并返回字符串。
    """
    try:
        text = bytes.fromhex(hex_str).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as e:
        print(f"解码错误: {e}")
        return ""
    print(f"十六进制 → 字符串: {text}")
    return text


if __name__ == "__main__":
    # 简单测试
    original = input("请输入")
    h = str_to_hex(original)
    recovered = hex_to_str(h)
    print(f"还原验证: {original == recovered}")