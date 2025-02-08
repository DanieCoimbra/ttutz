
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import nltk
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode  # Remove acentos

# Baixar recursos do NLTK
nltk.download('punkt')#Tokeniza sentenças em palavras.
nltk.download('wordnet')#Permite a lemmatização, reduzindo palavras à sua forma básica.

lemmatizer = WordNetLemmatizer()

# Base de conhecimento
study_routines = {
    "algebra linear": "Estude teoria com 'Álgebra Linear - Gilbert Strang', resolva exercícios diariamente e utilize recursos visuais como gráficos para melhor compreensão. Inclua tópicos como Vetores e Espaços Vetoriais, Matrizes e Determinantes, Sistemas Lineares, Autovalores e Autovetores, e Transformações Lineares e revisão.",
    "banco de dados": "Leia 'Introdução a sistemas de bancos de dados - DATE, C. J', pratique consultas SQL e implemente pequenos projetos para fixação. Estude tópicos como Introdução a Bancos de Dados, Arquiteturas de Banco de Dados, Modelagem Conceitual de Dados, Documentação de Modelos de Dados, Modelagem Lógica de Dados, Introdução ao SQL e ao Ambiente de banco de dados, Data Definition Language (DDL), Data Manipulation Language (DML) e Data Query Language (DQL).",
    "calculo": "Estude teoria com 'Cálculo - James Stewart', resolva muitos exercícios e utilize softwares como Wolfram Alpha para visualização. Inclua tópicos como Limites e Continuidade, Derivadas, Integrais, Séries Infinitas, Revisão e exercícios.",
    "circuitos eletricos": "Estude teoria e prática com livros e recursos visuais, como 'Introdução à análise de circuitos - Robert L. Boylestad'. Inclua tópicos como Circuitos em Corrente Contínua - Circuitos em Série, Circuitos em Paralelo, Métodos de Análise, Teoremas da Análise de Circuitos, Circuitos RC e RL, Circuitos Magnéticos, Circuitos em Corrente Alternada - Correntes e Tensões Alternadas Senoidais, Circuitos de CA Série e Paralelo em Regime Permanente e Circuitos RLC.",
    "eletromagnetismo": "Leia livros como 'Eletromagnetismo - Joseph Edminister; Mahmood Nahvi' e faça atividades práticas de laboratório. Inclua tópicos como Álgebra Vetorial, Campos Elétricos e Magnéticos, Eletrodinâmica, Materiais Dielétricos e Magnéticos, Propagação de Ondas Eletromagnéticas e Atividades de Laboratório.",
    "engenharia de software": "Leia 'Princípios de análise e projeto de sistemas com UML - BEZERRA, Eduardo', pratique modelagem e implemente pequenos sistemas usando boas práticas. Inclua tópicos como Visão Geral do processo de desenvolvimento de Software, Ciclos de vida de Software, Engenharia de Requisitos – Elicitação, Engenharia de Requisitos – Especificação, Análise Orientada a Objetos - Modelagem de classes de análise, Análise Orientada a Objetos - Modelagem de interações, Análise Orientada a Objetos - Modelagem de estados e Análise Orientada a Objetos - Modelagem de atividades.",
    "estatistica": "Estude teoria e pratique exercícios com livros como 'Estatística Básica - Wilton de Oliveira Bussab'. Inclua tópicos como Distribuições de Probabilidade, Inferência Estatística, Regressão e Correlação, Testes de Hipóteses, Análise de Variância e revisão.",
    "estrutura de computadores": "Leia livros como 'Arquitetura de sistemas operacionais - Francis Berenger Machado, Luiz Paulo Maia' e faça estudos práticos. Inclua tópicos como História do Computador, Principais dispositivos de hardware de um computador, Noções de redes de computadores: protocolos, topologias e cabeamento estruturado, Introdução aos Sistemas Operacionais, Principais subsistemas que compõem um sistema operacional, Máquinas virtuais, Sistema operacional Linux e Software Livre, e Ambientes gráficos e orientados a caractere do Linux.",
    "estrutura de dados": "Estude teoria e prática com livros como 'Algoritmos - Thomas H. Cormen'. Inclua tópicos como Conceito de Tipos Abstratos de Dados, Algoritmos e Estruturas de Dados, Abstração de Dados, Tipos de Dados e Tipos Estruturados de Dados, Lista Contígua, Apontadores, Lista Encadeada, Pilhas, Filas e Tabelas Hash, Árvores, Métodos de Ordenação e Métodos de Pesquisa.",
    "inteligencia artificial": "Leia 'Artificial intelligence: a modern approach - Stuart J. Russell; Peter Norvig' e pratique com linguagens como PROLOG, LISP e Java. Inclua tópicos como Histórico e visão geral da área da Inteligência Artificial, Problemas e espaço de estado, Técnicas de busca: desinformada e heurística, Representação e uso do conhecimento, Regras, objetos e lógica, Casamento de padrões, Processamento de Linguagem Natural, Robótica, Redes Neurais Artificiais, Sistemas Especialistas, Computação Evolutiva e Aprendizado Indutivo.",
    "praticas na engenharia": "Estude e aplique metodologias práticas com livros e recursos visuais. Inclua tópicos como O que é um projeto, Metodologia 5W2H, Metodologia Kanban, Ferramentas fundamentais do Excel para gerenciamento de projeto e Metodologia SWOT.",
    "programaçao orientada a objetos": "Estude conceitos de orientação a objetos com C# usando livros como 'Use a cabeça: C# - STELLMAN, Andrew; GREENE, Jennifer'. Inclua tópicos como Conceitos, definições e relacionamentos da Orientação a Objetos com C#, Coleções de dados em C#, Trabalhando com elementos visuais, Integrando Banco de Dados com aplicações desenvolvidas em C# e Padrão de Projeto.",
}


book_recommendations = {
    "algebra linear": ["Álgebra Linear - Gilbert Strang", "Introdução à Álgebra Linear - Howard Anton"],
    "banco de dados": ["Introdução a sistemas de bancos de dados - DATE, C. J", "Sistemas de banco de dados - ELMASRI, Ramez; NAVATHE, Sham.", "Sistema de banco de dados - Silberschatz, Abraham; Korth, Henry F.; Sudarshan, S."],
    "calculo": ["Cálculo - James Stewart", "Cálculo I - Elon Lages Lima"],
    "circuitos eletricos": ["Introdução à análise de circuitos - Robert L. Boylestad", "Fundamentos de circuitos elétricos - Charles K. Alexander; Matthew N. O. Sadiku", "Teoria e problemas de circuitos elétricos - Mahmood Nahvi; Joseph Edminister", "Circuitos elétricos - James William Nilsson; Susan A. Riedel"],
    "eletromagnetismo": ["Eletromagnetismo - Joseph Edminister; Mahmood Nahvi", "Eletromagnetismo para engenheiros - Clayton R. Paul", "Elementos de eletromagnetismo - Matthew N. O. Sadiku", "Fundamentos de eletromagnetismo com aplicações em engenharia - Stuart M. Wentworth"],
    "engenharia de software": ["Princípios de análise e projeto de sistemas com UML - BEZERRA, Eduardo", "Fundamentos do desenho orientado a objeto com UML - PAGE-JONES, Meilir", "Engenharia de software: uma abordagem profissional - PRESSMAN, Roger S.; MAXIM, Bruce R."],
    "estatistica": ["Estatística Básica - Wilton de Oliveira Bussab", "Probabilidade e Estatística - William W. Hines"],
    "estrutura de computadores": ["Arquitetura de sistemas operacionais - Francis Berenger Machado, Luiz Paulo Maia", "Redes de computadores e a internet: uma abordagem top-down - James F. Kurose, Keith W. Ross", "Sistemas operacionais modernos - Andrew S. Tanenbaum"],
    "estrutura de dados": ["Algoritmos - Thomas H. Cormen", "Algoritmos em linguagem C - Paulo Feofiloff", "Estruturas de dados: conceitos e técnicas de implementação - Marcos Vianna Villas", "Introdução a estruturas de dados: com técnicas de programação em C - Waldemar Celes, Renato Cerqueira, José Lucas Rangel"],
    "inteligencia artificial": ["Artificial intelligence: a modern approach - Stuart J. Russell; Peter Norvig", "Inteligência artificial - Stuart J. Russell; Peter Norvig", "Inteligência artificial: ferramentas e teorias - Guilherme Bittencourt", "Fundamentos matemáticos para a ciência da computação: um tratamento moderno de matemática discreta - Judith L. Gersting", "Inteligência artificial: estruturas e estratégias para a resolução de problemas complexos - George F. Luger"],
    "praticas na engenharia": ["Gerenciamento de projetos: guia do profissional - Claudius Jordão, Marcus Possi, Volume 1", "Gerenciamento de projetos: guia do profissional - Elizabeth Borges, Marcus Possi, Volume 2", "Gerência de projetos: guia para o exame oficial do PMI - Kim Heldman"],
    "programaçao orientada a objetos": ["Orientação a objetos e SOLID para ninjas - ANICHE, Maurício", "Use a cabeça: C# - STELLMAN, Andrew; GREENE, Jennifer"]
}


study_content = {
    "algebra linear": ["Vetores e Espaços Vetoriais", "Matrizes e Determinantes", "Sistemas Lineares", "Autovalores e Autovetores", "Transformações Lineares e revisão"],
    "banco de dados": ["Introdução a Bancos de Dados", "Arquiteturas de Banco de Dados", "Modelagem Conceitual de Dados", "Documentação de Modelos de Dados", "Modelagem Lógica de Dados", "Introdução ao SQL e ao Ambiente de banco de dados", "Data Definition Language (DDL)", "Data Manipulation Language (DML)", "Data Query Language (DQL)"],
    "calculo": ["Limites e Continuidade", "Derivadas", "Integrais", "Séries Infinitas", "Revisão e exercícios"],
    "circuitos eletricos": ["Circuitos em Corrente Contínua - Circuitos em Série", "Circuitos em Paralelo", "Métodos de Análise", "Teoremas da Análise de Circuitos", "Circuitos RC e RL", "Circuitos Magnéticos", "Circuitos em Corrente Alternada - Correntes e Tensões Alternadas Senoidais", "Circuitos de CA Série e Paralelo em Regime Permanente", "Circuitos RLC"],
    "eletromagnetismo": ["Álgebra Vetorial", "Campos Elétricos e Magnéticos", "Eletrodinâmica", "Materiais Dielétricos e Magnéticos", "Propagação de Ondas Eletromagnéticas", "Atividades de Laboratório"],
    "engenharia de software": ["Visão Geral do processo de desenvolvimento de Software", "Ciclos de vida de Software", "Engenharia de Requisitos – Elicitação", "Engenharia de Requisitos – Especificação", "Análise Orientada a Objetos - Modelagem de classes de análise", "Análise Orientada a Objetos - Modelagem de interações", "Análise Orientada a Objetos - Modelagem de estados", "Análise Orientada a Objetos - Modelagem de atividades"],
    "estatistica": ["Distribuições de Probabilidade", "Inferência Estatística", "Regressão e Correlação", "Testes de Hipóteses", "Análise de Variância e revisão"],
    "estrutura de computadores": ["História do Computador", "Principais dispositivos de hardware de um computador", "Noções de redes de computadores: protocolos, topologias e cabeamento estruturado", "Introdução aos Sistemas Operacionais", "Principais subsistemas que compõem um sistema operacional", "Máquinas virtuais", "Sistema operacional Linux e Software Livre", "Ambientes gráficos e orientados a caractere do Linux"],
    "estrutura de dados": ["Conceito de Tipos Abstratos de Dados", "Algoritmos e Estruturas de Dados", "Abstração de Dados", "Tipos de Dados e Tipos Estruturados de Dados", "Lista Contígua", "Apontadores", "Lista Encadeada", "Pilhas, Filas e Tabelas Hash", "Árvores", "Métodos de Ordenação", "Métodos de Pesquisa"],
    "inteligencia artificial": ["Histórico e visão geral da área da Inteligência Artificial", "Problemas e espaço de estado", "Técnicas de busca: desinformada e heurística", "Representação e uso do conhecimento", "Regras, objetos e lógica", "Casamento de padrões", "Uso de PROLOG, LISP e Java para tratar problemas de IA", "Processamento de Linguagem Natural", "Robótica", "Redes Neurais Artificiais", "Sistemas Especialistas", "Computação Evolutiva", "Aprendizado Indutivo"],
    "praticas na engenharia": ["O que é um projeto", "Metodologia 5W2H", "Metodologia Kanban", "Ferramentas fundamentais do Excel para gerenciamento de projeto","Metodologia SWOT"],
    "programaçao orientada a objetos": ["Conceitos, definições e relacionamentos da Orientação a Objetos com C#", "Coleções de dados em C#", "Trabalhando com elementos visuais", "Integrando Banco de Dados com aplicações desenvolvidas em C#", "Padrão de Projeto"],
}


# Intenções do chatbot,Define intenções (intents), que representam os diferentes assuntos que o chatbot pode entender.
intents = {
    "intents": [
        {"tag": "banco de dados", "patterns": ["Me indique um livro de banco de dados", "Sugira algo sobre arquitetura de dados"],
         "responses": book_recommendations["banco de dados"] + [study_routines["banco de dados"]]},
        {"tag": "engenharia de software", "patterns": ["Me indique um livro de engenharia de software", "Sugira algo sobre desenvolvimento de Software"],
         "responses": book_recommendations["engenharia de software"] + [study_routines["engenharia de software"]]},
        {"tag": "algebra linear", "patterns": ["Quero aprender álgebra linear", "Me sugira livros sobre álgebra"],
         "responses": book_recommendations["algebra linear"] + [study_routines["algebra linear"]]},
        {"tag": "calculo", "patterns": ["Preciso estudar cálculo", "Me recomende um livro de cálculo"],
         "responses": book_recommendations["calculo"] + [study_routines["calculo"]]},
        {"tag": "estatistica", "patterns": ["Me indique um livro de estatística", "Sugira algo sobre inferência"],
         "responses": book_recommendations["estatistica"] + [study_routines["estatistica"]]},
        {"tag": "programaçao orientada a objetos", "patterns": ["Me indique um livro de programação orientada a objetos", "Sugira algo sobre padrões de projeto"],
         "responses": book_recommendations["programaçao orientada a objetos"] + [study_routines["programaçao orientada a objetos"]]},
        {"tag": "eletromagnetismo", "patterns": ["Qual é o conteúdo da Unidade I da disciplina de Eletromagnetismo?", "O que é avaliado no curso de Eletromagnetismo?"],
         "responses": book_recommendations["eletromagnetismo"] + [study_routines["eletromagnetismo"]]},
        {"tag": "estrutura de dados", "patterns": ["Qual é o conteúdo da Unidade I da disciplina de Estrutura de Dados?", "O que é avaliado no curso de Estrutura de Dados?"],
         "responses": book_recommendations["estrutura de dados"] + [study_routines["estrutura de dados"]]},
        {"tag": "estrutura de computadores", "patterns": ["Qual é o conteúdo da Unidade I da disciplina de Estrutura de Computadores?", "O que é avaliado no curso de Estrutura de Computadores?"],
         "responses": book_recommendations["estrutura de computadores"] + [study_routines["estrutura de computadores"]]},
        {"tag": "praticas na engenharia", "patterns": ["Qual é o conteúdo da Unidade I da disciplina de Práticas na Engenharia?", "O que é avaliado no curso de Práticas na Engenharia?"],
         "responses": book_recommendations["praticas na engenharia"] + [study_routines["praticas na engenharia"]]},
        {"tag": "circuitos eletricos", "patterns": ["Qual é o conteúdo da Unidade I da disciplina de Circuitos Elétricos I?", "O que é avaliado no curso de Circuitos Elétricos I?"],
         "responses": book_recommendations["circuitos eletricos"] + [study_routines["circuitos eletricos"]]},
        {"tag": "inteligencia artificial", "patterns": ["Qual é o conteúdo da Unidade I da disciplina de Inteligência Artificial?", "O que é avaliado no curso de Inteligência Artificial?"],
         "responses": book_recommendations["inteligencia artificial"] + [study_routines["inteligencia artificial"]]}
    ]
}

# Remove acentos e transforma tudo em letras minúsculas.
def normalize_text(text):
    return unidecode(text.lower())
    
# Processamento dos dados para treino da IA
# Cria listas para armazenar palavras e categorias.
words = []
classes = []
documents = []
ignore_words = ["?", "!", ",", "."]
# Tokeniza as frases em palavras. Remove acentos e caracteres indesejados. Armazena palavras, classes e documentos processados.
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        norm_pattern = unidecode(pattern.lower())
        word_list = nltk.word_tokenize(norm_pattern)
        words.extend(word_list)
        documents.append((word_list, unidecode(intent["tag"])))
        if unidecode(intent["tag"]) not in classes:
            classes.append(unidecode(intent["tag"]))
# Lematiza todas as palavras e remove duplicatas
words = sorted(set([lemmatizer.lemmatize(w) for w in words if w not in ignore_words]))
classes = sorted(set(classes))

# Criando dados de treinamento
# Converte frases em vetores binários, associa cada vetor com a classe correta.
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = [1 if w in [lemmatizer.lemmatize(word) for word in doc[0]] else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
# Embaralha os dados para evitar viés no aprendizado, separa as entradas (train_x) e saídas (train_y).
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Criando modelo de IA
model = Sequential([ # Sequential(): Modelo de rede neural.
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'), # 128 neurônios (ReLU) → Melhora aprendizado
    Dropout(0.5), # Dropout de 50% → Evita overfitting.
    Dense(64, activation='relu'), # 64 neurônios (ReLU).
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax') # Softmax na saída → Classificação das intenções.
])

# Compilação com categorical_crossentropy (para classificação), Treinamento por 200 vezes.
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Função para gerar respostas do chatbot
# Converte a entrada do usuário em um vetor de palavras.
# Usa a rede neural para prever a intenção.
# Retorna uma resposta com pelo menos 70% de certeza.
def get_response(user_input):
    user_input = unidecode(user_input.lower())
    bag = [1 if w in [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(user_input)] else 0 for w in words]
    results = model.predict(np.array([bag]))[0]
    max_index = np.argmax(results)

    if results[max_index] > 0.7:
        for intent in intents["intents"]:
            if intent["tag"] == classes[max_index]:
                return random.choice(intent["responses"])
    return "Desculpe, não entendi. Pode reformular sua pergunta?"
    
# Função para gerar a rotina de estudos
# Distribui horas de estudo para cada disciplina ao longo das semanas.
# Retorna um cronograma personalizado.
def generate_study_plan(available_hours, subjects, weeks):
    print(f"\n📢 Gerando plano de estudos para as matérias: {subjects}\n")

    total_hours = sum(available_hours.values())
    if total_hours == 0:
        return "Você precisa definir pelo menos algumas horas para estudar."

    study_plan = f"📚 **Rotina de Estudos para {weeks} semanas:**\n"

    # Criar estrutura para armazenar os tópicos de cada matéria por semana
    weekly_topics = {subject: [[] for _ in range(weeks)] for subject in subjects}

    # Distribuir os tópicos de estudo uniformemente ao longo das semanas
    for subject in subjects:
        if subject in study_content:
            topics = study_content[subject]
            for i, topic in enumerate(topics):
                weekly_topics[subject][i % weeks].append(topic)
        else:
            print(f"⚠️ Aviso: Matéria '{subject}' não encontrada em study_content!")

    # Criando a rotina de estudos
    for week in range(weeks):
        study_plan += f"\n🗓️ **Semana {week + 1}**:\n"
        for day, hours in available_hours.items():
            if hours == 0:
                study_plan += f"\n  **{day}**:\n   - Dia de descanso\n"
            else:
                study_plan += f"\n  **{day}**:\n"
                if len(subjects) > 0:
                    time_per_subject = hours / len(subjects)
                else:
                    time_per_subject = 0  # Evitar divisão por zero
                
                for subject in subjects:
                    study_plan += f"   - **{subject.capitalize()}**: {time_per_subject:.1f} horas\n"
                    for topic in weekly_topics[subject][week]:
                        study_plan += f"     - 📌 {topic}\n"

    return study_plan

# Chatbot interativo
# O chatbot aceita perguntas ou gera um plano de estudos.
def chatbot():
    print("\nDigite 'rotina' para criar um plano de estudos ou faça uma pergunta sobre matérias!")
    while True:
        user_input = input("\nVocê: ")

        if user_input.lower() == "sair":
            break

        if user_input.lower() == "rotina":
            user_subjects = input("Quais matérias você quer estudar? (Separe por vírgulas): ").split(",")
            user_subjects = [normalize_text(subject.strip()) for subject in user_subjects]# Remover espaços extras e normalizar o texto
            valid_subjects = [subject for subject in user_subjects if subject in study_routines]# Filtrar matérias que existem no study_routines

            # Verificar se capturamos todas as matérias corretamente
            print(f"📢 Matérias reconhecidas: {valid_subjects}")

            if not valid_subjects:
                print("Chatbot: Não reconheci nenhuma matéria válida.")
                continue


            try:
                weeks = int(input("\nPor quantas semanas deseja distribuir os estudos? "))
                if weeks <= 0:
                    raise ValueError
            except ValueError:
                print("Erro: Digite um número válido de semanas.")
                continue

            available_hours = {}
            days = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
            print("\nDigite quantas horas você pode estudar por dia:")
            for day in days:
                try:
                    available_hours[day] = float(input(f"{day}: "))
                except ValueError:
                    available_hours[day] = 0  # Se o usuário errar, assume 0

            # Gerando e exibindo a rotina de estudos corrigida
            study_plan = generate_study_plan(available_hours, valid_subjects, weeks)
            print("\nChatbot:\n", study_plan)

        else:
            print("\nChatbot: Não entendi. Digite 'rotina' para criar um plano de estudos ou 'sair' para encerrar.")

#teste: algebra linear, banco de dados, engenharia de software, calculo, inteligencia artificial




if __name__ == "__main__":
    chatbot()
