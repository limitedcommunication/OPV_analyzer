from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib.pyplot as plt

def visualize_ecfp_bit(smiles, bit_to_visualize, radius=2, n_bits=1024):
    """
    可视化特定ECFP位对应的分子子结构（优化版）
    
    参数:
        smiles: 分子SMILES字符串
        bit_to_visualize: 要可视化的ECFP位(如904)
        radius: ECFP半径(默认2)
        n_bits: 指纹长度(默认1024)
    """
    # 1. 生成分子对象（增加SMILES解析容错）
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("错误：无效的SMILES字符串")
        return None
    
    try:
        # 添加原子索引标记便于识别
        for atom in mol.GetAtoms():
            atom.SetProp("atomNote", str(atom.GetIdx()))
    except Exception as e:
        print(f"原子标记错误: {e}")

    # 2. 生成指纹并获取位信息
    bit_info = {}
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 
            radius=radius, 
            nBits=n_bits, 
            bitInfo=bit_info
        )
    except Exception as e:
        print(f"指纹生成错误: {e}")
        return None
    
    # 3. 检查目标位是否被设置
    if bit_to_visualize not in bit_info:
        print(f"位 {bit_to_visualize} 在此分子中未激活")
        # 显示所有激活的位供参考
        print(f"\n当前分子激活的ECFP位示例: {list(bit_info.keys())[:10]}...")
        return None
    
    # 4. 获取该位对应的所有环境
    print(f"\n位 {bit_to_visualize} 对应的子结构信息:")
    for i, (atom_idx, rad) in enumerate(bit_info[bit_to_visualize]):
        print(f"  环境 {i+1}: 中心原子索引 {atom_idx} ({mol.GetAtomWithIdx(atom_idx).GetSymbol()}), 半径 {rad}")
        
        # 5. 提取该环境对应的原子和键
        try:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
            atoms_to_highlight = set()
            for bond_idx in env:
                bond = mol.GetBondWithIdx(bond_idx)
                atoms_to_highlight.add(bond.GetBeginAtomIdx())
                atoms_to_highlight.add(bond.GetEndAtomIdx())
            
            # 添加中心原子（半径=0时env可能为空）
            atoms_to_highlight.add(atom_idx)
            
            # 6. 优化可视化
            img = Draw.MolToImage(
                mol, 
                highlightAtoms=list(atoms_to_highlight), 
                highlightBonds=list(env),
                highlightColor=(0.8, 0.2, 0.2),  # 更醒目的红色
                size=(800, 800),  # 增大图像尺寸
                kekulize=True,
                wedgeBonds=True
            )
            
            # 显示带原子索引的分子
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.title(f'ECFP位 {bit_to_visualize}\n环境 {i+1} (中心原子 {atom_idx}, 半径 {rad})', 
                     fontsize=12, pad=20)
            plt.axis('off')
            
            # 添加说明文本
            center_atom = mol.GetAtomWithIdx(atom_idx)
            plt.text(0.5, 0.05, 
                    f"中心原子: {center_atom.GetSymbol()}({atom_idx}), 总键数: {center_atom.GetDegree()}", 
                    transform=plt.gca().transAxes,
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8))
            
            plt.show()
            
            # 打印子结构SMARTS
            if rad > 0:
                submol = Chem.PathToSubmol(mol, env)
                print(f"    SMARTS表示: {Chem.MolToSmarts(submol)}")
            
        except Exception as e:
            print(f"环境 {i+1} 可视化失败: {e}")
    
    return bit_info

# 使用示例
if __name__ == "__main__":
    # 您的复杂分子
    smiles = "CCCCCCCCC(CN(C1=C2C3=NC4=CC=CC5=CC=CC(N=C3C6=C1N(CC(CCCCCCCC)CCCCCC)C7=C6SC8=C7SC(/C=C9/C(C%10=C(C=C(C(F)=C%10)F)C9=O)=C(C#N)/C#N)=C8CCCCCCCCCCC)=C54)C%11=C2SC%12=C%11SC(/C=C%13/C(C%14=C(C=C(C(F)=C%14)F)C%13=O)=C(C#N)/C#N)=C%12CCCCCCCCCCC)CCCCCC"
    
    # 需要分析的位列表
    #target_bits = [904,552, 392, 425, 561, 277] pos_average
    target_bits = [425, 440, 530, 832, 13, 428]
    
    for bit in target_bits:
        print(f"\n{'='*50}")
        print(f"分析ECFP位: {bit}")
        print(f"{'='*50}")
        visualize_ecfp_bit(smiles, bit)
        print(f"{'='*50}\n")