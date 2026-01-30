import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import subprocess
import os
import re

class AlephFraudDetector:
    """Détecteur de fraude utilisant Aleph (ILP) pour apprendre des règles symboliques"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.learned_rules = []
        
    def load_data(self, df, sample_size=3000, fraud_ratio=0.20):  
        """Charge les données avec contrôle du ratio de fraude"""
        frauds = df[df['Is Laundering'] == 1]
        legit = df[df['Is Laundering'] == 0]
        
        n_fraud = min(int(sample_size * fraud_ratio), len(frauds))
        n_legit = sample_size - n_fraud
        
        print(f"🎯 Configuration:")
        print(f"   Sample size: {sample_size}")
        print(f"   Fraud ratio: {fraud_ratio*100:.0f}%")
        print(f"   → {n_fraud} fraudes + {n_legit} légitimes")
        
        fraud_sample = frauds.sample(n=n_fraud, random_state=42)
        legit_sample = legit.sample(n=n_legit, random_state=42)
        
        self.df = pd.concat([fraud_sample, legit_sample], ignore_index=True).sample(frac=1, random_state=42)
        
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        self.df['From_Account'] = self.df['From Bank'].astype(str) + '_' + self.df['Account'].astype(str)
        self.df['To_Account'] = self.df['To Bank'].astype(str) + '_' + self.df['Account.1'].astype(str)
        
        print(f"✓ {len(self.df)} transactions chargées")
        print(f"✓ Fraudes: {self.df['Is Laundering'].sum()} ({self.df['Is Laundering'].mean()*100:.2f}%)")
        
    def build_graph(self):
        """Construit le graphe de transactions"""
        print("\n📊 Construction du graphe...")
        
        for idx, row in self.df.iterrows():
            self.graph.add_edge(
                row['From_Account'],
                row['To_Account'],
                amount=row['Amount Paid'],
                timestamp=row['Timestamp'],
                is_laundering=row['Is Laundering'],
                transaction_id=idx
            )
        
        print(f"✓ Graphe: {self.graph.number_of_nodes()} comptes, {self.graph.number_of_edges()} transactions")
    
    def extract_features(self):
        """Extrait des features du graphe pour chaque transaction"""
        print("\n🔍 Extraction des features du graphe...")
        
        features = []
        
        for idx, row in self.df.iterrows():
            from_acc = row['From_Account']
            to_acc = row['To_Account']
            
            out_degree = self.graph.out_degree(from_acc)
            in_degree = self.graph.in_degree(from_acc)
            

            to_out_degree = self.graph.out_degree(to_acc)
            to_in_degree = self.graph.in_degree(to_acc)
            
            amount = row['Amount Paid']
            amount_category = 'low' if amount < 5000 else 'medium' if amount < 10000 else 'high'

            is_round = 'yes' if amount % 1000 == 0 else 'no'

            payment_type = row['Payment Format'].lower().replace(' ', '_')

            in_cycle = 'no'
            try:
                if nx.has_path(self.graph, to_acc, from_acc):
                    path_length = nx.shortest_path_length(self.graph, to_acc, from_acc)
                    if path_length <= 3:
                        in_cycle = 'yes'
            except:
                pass

            degree_category = 'low' if out_degree < 5 else 'medium' if out_degree < 20 else 'high'
            
            features.append({
                'transaction_id': idx,
                'out_degree': out_degree,
                'in_degree': in_degree,
                'to_out_degree': to_out_degree,
                'to_in_degree': to_in_degree,
                'amount_category': amount_category,
                'is_round_amount': is_round,
                'payment_type': payment_type,
                'in_cycle': in_cycle,
                'degree_category': degree_category,
                'is_laundering': row['Is Laundering']
            })
        
        self.features_df = pd.DataFrame(features)
        print(f"✓ {len(features)} transactions avec features extraites")
        return self.features_df
    
    def generate_prolog_file(self):
        """Génère le fichier Prolog pour Aleph"""
        print("\n📝 Génération du fichier Prolog pour Aleph...")

        frauds = self.features_df[self.features_df['is_laundering'] == 1]
        legit = self.features_df[self.features_df['is_laundering'] == 0]

        fraud_train = frauds.sample(frac=0.7, random_state=42)
        fraud_test = frauds.drop(fraud_train.index)
        
        legit_train = legit.sample(frac=0.7, random_state=42)
        legit_test = legit.drop(legit_train.index)
        
        train_df = pd.concat([fraud_train, legit_train])
        test_df = pd.concat([fraud_test, legit_test])
        
        with open('fraud_detection.pl', 'w') as f:
            f.write(":- use_module('aleph/prolog/aleph.pl').\n")
            f.write(":- aleph.\n\n")
            
            
            f.write("% Aleph settings\n")
            f.write(":- aleph_set(verbosity, 1).\n")
            f.write(":- aleph_set(noise, 100).\n")          
            f.write(":- aleph_set(evalfn, coverage).\n")    
            f.write(":- aleph_set(nodes, 200000).\n")       
            f.write(":- aleph_set(clauselength, 3).\n")     
            f.write(":- aleph_set(depth, 10).\n")
            f.write(":- aleph_set(minpos, 3).\n")           
            f.write(":- aleph_set(minacc, 0.05).\n\n")      

            f.write("% Mode declarations\n")
            f.write(":- modeh(1, laundering(+transaction)).\n")
            f.write(":- modeb(*, has_high_out_degree(+transaction)).\n")
            f.write(":- modeb(*, has_high_in_degree(+transaction)).\n")
            f.write(":- modeb(*, high_amount(+transaction)).\n")
            f.write(":- modeb(*, medium_amount(+transaction)).\n")
            f.write(":- modeb(*, is_round_amount(+transaction)).\n")
            f.write(":- modeb(*, in_cycle(+transaction)).\n")
            f.write(":- modeb(*, payment_type(+transaction, #payment)).\n")
            f.write(":- modeb(*, to_has_high_degree(+transaction)).\n\n")

            f.write("% Determinations\n")
            f.write(":- determination(laundering/1, has_high_out_degree/1).\n")
            f.write(":- determination(laundering/1, has_high_in_degree/1).\n")
            f.write(":- determination(laundering/1, high_amount/1).\n")
            f.write(":- determination(laundering/1, medium_amount/1).\n")
            f.write(":- determination(laundering/1, is_round_amount/1).\n")
            f.write(":- determination(laundering/1, in_cycle/1).\n")
            f.write(":- determination(laundering/1, payment_type/2).\n")
            f.write(":- determination(laundering/1, to_has_high_degree/1).\n\n")

            f.write("% Positive examples (frauds)\n")
            f.write(":- begin_in_pos.\n")
            for _, row in fraud_train.iterrows():
                f.write(f"laundering(t{row['transaction_id']}).\n")
            f.write(":- end_in_pos.\n\n")

            f.write("% Negative examples (legitimate)\n")
            f.write(":- begin_in_neg.\n")
            for _, row in legit_train.iterrows():
                f.write(f"laundering(t{row['transaction_id']}).\n")
            f.write(":- end_in_neg.\n\n")

            f.write("% Background knowledge\n")
            f.write(":- begin_bg.\n")
            for _, row in train_df.iterrows():
                tid = f"t{row['transaction_id']}"
                
                if row['out_degree'] >= 10:
                    f.write(f"has_high_out_degree({tid}).\n")
                if row['in_degree'] >= 10:
                    f.write(f"has_high_in_degree({tid}).\n")
                if row['to_out_degree'] >= 10 or row['to_in_degree'] >= 10:
                    f.write(f"to_has_high_degree({tid}).\n")
                if row['amount_category'] == 'high':
                    f.write(f"high_amount({tid}).\n")
                if row['amount_category'] == 'medium':
                    f.write(f"medium_amount({tid}).\n")
                if row['is_round_amount'] == 'yes':
                    f.write(f"is_round_amount({tid}).\n")
                if row['in_cycle'] == 'yes':
                    f.write(f"in_cycle({tid}).\n")
                f.write(f"payment_type({tid}, {row['payment_type']}).\n")
            f.write(":- end_bg.\n")

        test_df.to_csv('test_set.csv', index=False)
        
        print(f"✓ Fichier Prolog généré: fraud_detection.pl")
        print(f"✓ Train: {len(train_df)} transactions ({len(fraud_train)} fraudes, {len(fraud_train)/len(train_df)*100:.1f}%)")
        print(f"✓ Test: {len(test_df)} transactions ({len(fraud_test)} fraudes, {len(fraud_test)/len(test_df)*100:.1f}%)")
        
        return train_df, test_df
    
    def run_aleph(self):
        """Exécute Aleph pour apprendre des règles"""
        print("\n🤖 Lancement d'Aleph (apprentissage des règles, peut prendre quelques minutes)...")
        
        with open('run_aleph.pl', 'w') as f:
            f.write(":- use_module('aleph/prolog/aleph.pl').\n")
            f.write(":- initialization(main).\n\n")
            f.write("main :-\n")
            f.write("\twriteln('>>> RUN_ALEPH.PL DEMARRE <<<'),\n")
            f.write("\tconsult('fraud_detection.pl'),\n")
            f.write("\tinduce,\n")
            f.write("\ttell('learned_rules.pl'),\n")
            f.write("\tshow(theory),\n")
            f.write("\ttold,\n")
            f.write("\twriteln('>>> RUN_ALEPH.PL TERMINE <<<'),\n")
            f.write("\thalt.\n")

        try:
            result = subprocess.run(
                [r'swipl/bin/swipl.exe',
                 '-q',
                 '-f', 'run_aleph.pl'],
                capture_output=True,
                text=True,
            )

            print("✓ Aleph terminé !")
            
            print("--- FIN SORTIE ---\n")
            
            if os.path.exists('learned_rules.pl'):
                with open('learned_rules.pl', 'r') as f:
                    rules = f.read()
                if rules.strip():
                    print("\n📚 RÈGLES APPRISES PAR ALEPH:")
                    print("="*60)
                    print(rules)
                    print("="*60)
                    self.learned_rules = rules
                    self._parse_rules(rules)
                else:
                    print("⚠️ Fichier learned_rules.pl vide - AUCUNE RÈGLE APPRISE")
                    self.learned_rules = None
            else:
                print("⚠️ Fichier learned_rules.pl non créé")
                self.learned_rules = None
                
        except subprocess.TimeoutExpired:
            print("⚠️ Timeout - Aleph a pris trop de temps")
        except Exception as e:
            print(f"❌ Erreur lors de l'exécution d'Aleph: {e}")
    
    def _parse_rules(self, rules_text):
        """Parse les règles Prolog apprises par Aleph"""
        self.parsed_rules = []
        

        pattern = r'laundering\(([A-Z])\)\s*:-\s*(.+?)\.'
        matches = re.findall(pattern, rules_text, re.DOTALL)
        
        print(f"\n🔍 Parsing {len(matches)} règle(s) générale(s)...")
        
        for var, body in matches:
            conditions = []
            current = ""
            paren_depth = 0
            
            for char in body:
                if char == '(':
                    paren_depth += 1
                    current += char
                elif char == ')':
                    paren_depth -= 1
                    current += char
                elif char == ',' and paren_depth == 0:
                    if current.strip():
                        conditions.append(current.strip())
                    current = ""
                else:
                    current += char
            
            
            if current.strip():
                conditions.append(current.strip())
            
            if conditions:  
                self.parsed_rules.append(conditions)
                print(f"  ✓ Règle {len(self.parsed_rules)}: {' AND '.join(conditions)}")
    
    def _apply_rule(self, row, conditions):
        """Applique une règle Prolog à une transaction"""
        for condition in conditions:
            if 'has_high_out_degree' in condition and row['out_degree'] < 10:
                return False
            
            if 'has_high_in_degree' in condition and row['in_degree'] < 10:
                return False
            
            if 'high_amount' in condition and row['amount_category'] != 'high':
                return False
            
            if 'medium_amount' in condition and row['amount_category'] != 'medium':
                return False
            
            if 'is_round_amount' in condition and row['is_round_amount'] != 'yes':
                return False
            
            if 'in_cycle' in condition and row['in_cycle'] != 'yes':
                return False
            
            if 'to_has_high_degree' in condition:
                if row['to_out_degree'] < 10 and row['to_in_degree'] < 10:
                    return False
            
            if 'payment_type' in condition:
                match = re.search(r'payment_type\([^,]+,([^)]+)\)', condition)
                if match:
                    required_type = match.group(1).strip()
                    if row['payment_type'] != required_type:
                        return False
                else:
                    return False
        
        return True
    
    def evaluate_rules(self):
        """Évalue les règles apprises sur le test set"""
        print("\n📊 Évaluation des règles apprises...")
        
        test_df = pd.read_csv('test_set.csv')
        
        if not self.learned_rules or self.learned_rules.strip() == "":
            print("\n⚠️ AUCUNE RÈGLE APPRISE PAR ALEPH !")
            test_df['predicted'] = 0
            
        elif hasattr(self, 'parsed_rules') and len(self.parsed_rules) > 0:
            print(f"\n✓ Application de {len(self.parsed_rules)} règle(s) apprise(s)...")
            
            test_df['predicted'] = 0
            for idx, row in test_df.iterrows():
                for rule_conditions in self.parsed_rules:
                    if self._apply_rule(row, rule_conditions):
                        test_df.loc[idx, 'predicted'] = 1
                        break
        else:
            print("\n⚠️ Impossible de parser les règles, prédictions à 0")
            test_df['predicted'] = 0

        tp = ((test_df['is_laundering'] == 1) & (test_df['predicted'] == 1)).sum()
        fp = ((test_df['is_laundering'] == 0) & (test_df['predicted'] == 1)).sum()
        tn = ((test_df['is_laundering'] == 0) & (test_df['predicted'] == 0)).sum()
        fn = ((test_df['is_laundering'] == 1) & (test_df['predicted'] == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / len(test_df)
        
        print("\n" + "="*60)
        print("📈 RÉSULTATS DE L'IA SYMBOLIQUE (ALEPH)")
        print("="*60)
        print(f"Précision: {precision:.2%} (si 0%, aucune règle n'a été apprise)")
        print(f"Rappel: {recall:.2%}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"\nVrais positifs: {tp}")
        print(f"Faux positifs: {fp}")
        print(f"Vrais négatifs: {tn}")
        print(f"Faux négatifs: {fn}")
        
        if tp == 0 and fp == 0:
            print("\n⚠️ DIAGNOSTIC: Aucune fraude détectée")
        
        print("="*60)
    
    def train(self):
        """Pipeline complet d'apprentissage"""
        print("\n" + "="*60)
        print("🎓 APPRENTISSAGE SYMBOLIQUE AVEC ALEPH (ILP)")
        print("="*60)
        
        self.build_graph()
        self.extract_features()
        self.generate_prolog_file()
        self.run_aleph()
        self.evaluate_rules()


if __name__ == "__main__":
    df_li = pd.read_csv('LI-Small_Trans.csv')
    df_hi = pd.read_csv('HI-Small_Trans.csv')
    df = pd.concat([df_li, df_hi], ignore_index=True)
    
    detector = AlephFraudDetector()
    
    detector.load_data(df, sample_size=30000, fraud_ratio=0.40)  
    
    detector.train()
    
    print("\n✅ Apprentissage terminé !")