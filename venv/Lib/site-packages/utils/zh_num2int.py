import re
import logging
import traceback

logger = logging.getLogger(__name__)


def zh_upper_num2int(strs: str):
    '''
        把中文数字转换为阿拉伯数字
        eg: 一千五百五十 -> 1550
        :param strs:
        :return:
        '''
    result = 0
    temp = 1  # 存放一个单位的数字如：十万
    count = 0  # 判断是否有chArr
    cn_dict = {'〇': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '零': 0,
               '壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5, '陆': 6, '柒': 7, '捌': 8, '玖': 9, '貮': 2, '两': 2, }
    chArr = ['十', '百', '千', '万', '亿']
    for i in range(len(strs)):
        b = True
        c = strs[i]
        if c in cn_dict:
            if count != 0:
                result += temp
                count = 0
            temp = cn_dict.get(c)
            b = False
            # break
        if b:
            for j in range(len(chArr)):
                if c == chArr[j]:
                    if j == 0:
                        temp *= 10
                    elif j == 1:
                        temp *= 100
                    elif j == 2:
                        temp *= 1000
                    elif j == 3:
                        temp *= 10000
                    elif j == 4:
                        temp *= 100000000
                count += 1
        if i == len(strs) - 1:
            result += temp

    return result


def date_num2int(strs):
    '''
    日期格式的中文大写数字转为阿拉伯数字
    eg: 一九九零年五月十三日 -> 1990年5月13日
    :param strs:
    :return:
    '''
    CN_NUM = {
        '〇': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '零': 0,
        '壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5, '陆': 6, '柒': 7, '捌': 8, '玖': 9, '貮': 2, '两': 2, }

    new_str = ''
    for i in strs:
        new_str += str(CN_NUM.get(i, i))
    return new_str


def zh_num2int(strs: str):
    '''

    :param strs: 一句话 eg: 我想要两百斤大米，一百二十斤小米。
    :return:
    '''
    try:
        pattrern2 = '[〇一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖貮两]{2,}'
        digits = [(strs[a.span()[0]:a.span()[1]], a.span()) for a in re.finditer(pattrern2, strs)]
        slide_num = 0
        chArr = ['十', '百', '千', '万', '亿']
        for d in digits:
            sub_str = d[0]
            sub_str_index = d[1]

            if re.search('|'.join(chArr), sub_str):
                # 若格式和二万二一样，则需要在后面补上具体的面值
                if sub_str[-1] in ['一', '二', '三', '四', '五', '六', '七', '八', '九']:
                    for i in range(len(sub_str) - 1, 0, -1):
                        if sub_str[i] in chArr:
                            index = chArr.index(sub_str[i])
                            if index == 0:
                                break
                            else:
                                sub_str = sub_str + chArr[index - 1]
                            break

                digit = zh_upper_num2int(sub_str)
            else:
                digit = date_num2int(sub_str)

            front = strs[:sub_str_index[0] + slide_num]
            after = strs[sub_str_index[1] + slide_num:]
            strs = front + str(digit) + after
            if len(str(digit)) != len(d[0]):
                slide_num += len(str(digit)) - len(d[0])
        return strs
    except Exception:
        logger.error(traceback.format_exc())
        return strs


if __name__ == '__main__':
    # print(zh_num2int('一九九零年五月十三日'))
    # print(zh_num2int('我想要两百斤大米，一百二十斤小米。'))
    print(zh_num2int('一百二'))
    print(zh_num2int('一百斤一千五十斤'))
    print(zh_num2int('一百一十一'))
    print(zh_num2int('一百斤一千五百五十斤'))
