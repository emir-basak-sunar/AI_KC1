# predict_defect.py

import re
import joblib
import pandas as pd

def analyze_c_code(code: str):
    lines = code.strip().split('\n')

    loc = len(lines)
    loc_blank = sum(1 for line in lines if line.strip() == '')
    loc_comment = sum(1 for line in lines if re.match(r'\s*//', line))
    loc_code = loc - loc_blank - loc_comment
    loc_code_and_comment = loc_code + loc_comment  # Bu satır, locCodeAndComment hesaplamasını yapar.

    mccabe_complexity = 1 + sum(len(re.findall(r'\b(if|for|while|case)\b', line)) for line in lines)

    operators = re.findall(r'[+\-*/=<>!&|]+|return|if|else|while|for|switch|case|break', code)
    operands = re.findall(r'\b[_a-zA-Z][_a-zA-Z0-9]*\b', code)

    keywords = {'return', 'if', 'else', 'while', 'for', 'switch', 'case', 'break', 'int', 'float', 'double', 'char', 'void', 'printf', 'main'}
    operands = [op for op in operands if op not in keywords]

    uniq_op = len(set(operators))
    uniq_opnd = len(set(operands))
    total_op = len(operators)
    total_opnd = len(operands)

    branch_count = sum(len(re.findall(r'\b(if|case|else if)\b', line)) for line in lines)

    # Ek metrikler ekleniyor
    v_g = loc_code  # Örnek: Kodun toplam satır sayısı
    ev_g = mccabe_complexity  # McCabe karmaşıklığı
    iv_g = loc_comment  # Yorum satırları
    n = len(set(operators))  # Operatör çeşitliliği
    v = len(set(operands))  # Operan çeşitliliği
    l = loc_code  # Kod satırları
    d = loc_blank  # Boş satırlar
    i = len(operands)  # Operan sayısı
    e = len(operators)  # Operatör sayısı
    b = branch_count  # Şart/branch sayısı
    t = loc_code_and_comment  # Kod ve yorum satırlarının toplamı

    return {
        "loc": loc,
        "lOCode": loc_code,
        "lOComment": loc_comment,
        "lOBlank": loc_blank,
        "locCodeAndComment": loc_code_and_comment,  # Burada locCodeAndComment özelliği eklendi
        "v(g)": v_g,
        "ev(g)": ev_g,
        "iv(g)": iv_g,
        "n": n,
        "v": v,
        "l": l,
        "d": d,
        "i": i,
        "e": e,
        "b": b,
        "t": t,
        "uniq_Op": uniq_op,
        "uniq_Opnd": uniq_opnd,
        "total_Op": total_op,
        "total_Opnd": total_opnd,
        "branchCount": branch_count,
    }



# Modeli yükle
model = joblib.load("trained_defect_model.pkl")

# Kullanıcıdan C kodu al
print("C kodunuzu girin (bitirmek için tek satıra sadece 'END' yazın):")
user_lines = []
while True:
    line = input()
    if line.strip() == 'END':
        break
    user_lines.append(line)

user_code = "\n".join(user_lines)

# Metrikleri çıkar
features = analyze_c_code(user_code)

# Modele uygun sırayla veriyi al
X_input = pd.DataFrame([[
    features['loc'],
    features['v(g)'],
    features['ev(g)'],
    features['iv(g)'],
    features['n'],
    features['v'],
    features['l'],
    features['d'],
    features['i'],
    features['e'],
    features['b'],
    features['t'],
    features['lOCode'],
    features['lOComment'],
    features['lOBlank'],
    features['locCodeAndComment'],  # Buradaki ismi doğru şekilde değiştirdik
    features['uniq_Op'],
    features['uniq_Opnd'],
    features['total_Op'],
    features['total_Opnd'],
    features['branchCount']
]], columns=[
    'loc',
    'v(g)',
    'ev(g)',
    'iv(g)',
    'n',
    'v',
    'l',
    'd',
    'i',
    'e',
    'b',
    't',
    'lOCode',
    'lOComment',
    'lOBlank',
    'locCodeAndComment',  # Burada da ismi değiştirdik
    'uniq_Op',
    'uniq_Opnd',
    'total_Op',
    'total_Opnd',
    'branchCount'
])


# Tahmin yap
prediction = model.predict(X_input)
print("\n🔍 Tahmin:", "Hatalı" if prediction[0] == 1 else "Hatasız")
