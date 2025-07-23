"""
Chudnovsky算法：
使用公式：1/π = 12Σ [(-1)^k * (6k)! * (545140134k + 13591409) / ((3k)! * (k!)^3 * 640320^(3k+1.5))]
通过数学变换将计算分解为整数运算，避免浮点精度损失。
多线程处理：
自动检测CPU核心数，并将级数计算任务分配到多个进程。
每个进程计算指定范围内的项，返回分数形式的结果（分子和分母）。
高精度计算：
使用牛顿迭代法计算sqrt(640320)到所需精度（整数运算）。
通过高精度除法获取π的小数部分。
"""
import math
import multiprocessing as mp
from functools import reduce

def compute_pi_terms(start, end):
    """计算Chudnovsky级数的指定范围内的项（分数形式）"""
    C = 640320**3  # 常数C，避免重复计算
    total_num = 0
    total_den = 1
    
    for k in range(start, end + 1):
        # 计算分子
        sign = 1 if k % 2 == 0 else -1
        numerator = sign * math.factorial(6 * k) * (545140134 * k + 13591409)
        
        # 计算分母
        denominator = math.factorial(3 * k) * (math.factorial(k))**3 * (C**k)
        
        # 将当前项加到总和中（通分）
        if numerator != 0:
            # 通分公式: a/b + c/d = (a*d + b*c) / (b*d)
            new_num = total_num * denominator + numerator * total_den
            new_den = total_den * denominator
            # 约分
            gcd_val = math.gcd(new_num, new_den)
            if gcd_val != 0:
                total_num = new_num // gcd_val
                total_den = new_den // gcd_val
            else:
                total_num, total_den = new_num, new_den
        else:
            # 当前项为0，跳过
            continue
    
    return total_num, total_den

def high_precision_division(numerator, denominator, digits):
    """高精度除法，计算分子/分母的小数部分（返回字符串）"""
    # 计算整数部分
    quotient = numerator // denominator
    remainder = numerator % denominator
    
    # 计算小数部分
    decimal_digits = []
    for _ in range(digits + 10):  # 多计算10位用于四舍五入
        if remainder == 0:
            break
        remainder *= 10
        digit = remainder // denominator
        decimal_digits.append(str(digit))
        remainder = remainder % denominator
    
    # 确保有足够的位数
    while len(decimal_digits) < digits + 1:
        decimal_digits.append('0')
    
    return ''.join(decimal_digits)

def main():
    # 确定计算参数
    precision_digits = 1000  # 目标精度（小数位数）
    extra_digits = 10       # 额外计算的位数（用于四舍五入）
    total_terms = 70        # 迭代项数（经验值，足够达到1000位精度）
    
    # 获取喵U核心数并分配任务
    num_cores = mp.cpu_count()
    chunk_size = (total_terms + num_cores - 1) // num_cores
    ranges = [
        (i * chunk_size, min((i + 1) * chunk_size - 1, total_terms))
        for i in range(num_cores)
    ]
    
    # 多进程计算级数项
    pool = mp.Pool(processes=num_cores)
    results = pool.starmap(compute_pi_terms, ranges)
    pool.close()
    pool.join()
    
    # 合并结果（所有块的通分求和）
    S_num, S_den = reduce(
        lambda x, y: (x[0] * y[1] + y[0] * x[1], x[1] * y[1]),
        results
    )
    S_gcd = math.gcd(S_num, S_den)
    if S_gcd != 0:
        S_num //= S_gcd
        S_den //= S_gcd
    
    # 计算常数 sqrt(640320) 到高精度
    sqrt_digits = precision_digits + extra_digits + 10  # 额外精度
    a_val = 640320 * 10**(2 * sqrt_digits)  # 缩放以进行整数平方根计算
    x0 = a_val
    x1 = (x0 + a_val // x0) // 2
    while x1 < x0:
        x0 = x1
        x1 = (x0 + a_val // x0) // 2
    sqrt_val = x0  # sqrt(640320 * 10^(2*sqrt_digits))
    
    # 计算 π = (640320 * sqrt(640320)) / (12 * S)
    # 分子: 640320 * sqrt_val * S_den
    numerator_val = 640320 * sqrt_val * S_den
    # 分母: 12 * S_num * 10^sqrt_digits
    denominator_val = 12 * S_num * (10**sqrt_digits)
    
    # 高精度除法获取小数部分
    decimal_str = high_precision_division(numerator_val, denominator_val, precision_digits + 1)
    
    # 格式化和四舍五入
    pi_decimal = decimal_str[:precision_digits]  # 取前1000位
    round_digit = int(decimal_str[precision_digits])  # 第1001位用于四舍五入
    
    # 处理四舍五入
    if round_digit >= 5:
        # 将小数部分转为整数并加1
        num_val = int(pi_decimal) + 1
        pi_decimal = str(num_val).zfill(precision_digits)
        # 处理进位溢出（如999...9变为1000...0）
        if len(pi_decimal) > precision_digits:
            pi_decimal = pi_decimal[1:]  # 移除前导进位（整数部分已处理）
    
    # 输出结果
    print("圆周率的前1000位（小数部分）：")
    print(f"3.{pi_decimal}")

if __name__ == '__main__':
    main()