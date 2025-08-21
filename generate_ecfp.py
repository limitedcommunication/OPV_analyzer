import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tkinter import Tk, filedialog
import os

def select_csv_file():
    """让用户选择CSV文件"""
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    file_path = filedialog.askopenfilename(
        title="选择包含SMILES和Pos_Average值的CSV文件",
        filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
    )
    root.destroy()
    return file_path

def smiles_to_ecfp(smiles, radius=2, n_bits=1024,useFeatures=True,useChirality=True):
    """将SMILES转换为ECFP指纹"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))

def generate_ecfp_features():
    # 1. 让用户选择CSV文件
    csv_path = select_csv_file()
    if not csv_path:
        print("未选择文件，程序退出")
        return None
    
    # 2. 读取CSV文件
    df = pd.read_csv(csv_path)
    print(f"成功读取文件: {os.path.basename(csv_path)}")
    print(f"数据前几行:\n{df.head()}")
    
    # 检查必要的列是否存在
    if 'smiles' not in df.columns or 'Pos_Average' not in df.columns:
        raise ValueError("CSV文件必须包含'smiles'和'Pos_Average'两列")
    
    # 3. 生成ECFP指纹
    print("正在生成ECFP指纹...")
    df['ECFP'] = df['smiles'].apply(lambda x: smiles_to_ecfp(x))
    
    # 移除无法生成指纹的行
    df = df.dropna(subset=['ECFP'])
    print(f"处理后数据量: {len(df)}")
    
    # 4. 保存结果到新CSV文件
    output_path = os.path.join(os.path.dirname(csv_path), 
                             f"ecfp_{os.path.basename(csv_path)}")
    df.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")
    
    return output_path

if __name__ == "__main__":
    generate_ecfp_features()