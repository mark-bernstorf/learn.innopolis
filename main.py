import pandas
import scipy
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Домашняя работа по теме "Статистика", "Визуализация", "Развертывание в виде веб приложения"')

st.caption(f'''
В качестве примера использую датасет с  оценками игр на метакритике.           
Попытаемся понять, есть ли связь между оценками критиков сервиса metacritics и оценками пользователей''')

uploaded_file = st.file_uploader("Выбор файла с датасетом (.csv)", type='csv')
if uploaded_file is not None:
    df = pandas.read_csv(uploaded_file)
else:
    exit(0)

st.dataframe(df)
st.markdown('Нужно выбрать две колонки из датасета')

col1, col2 = st.columns(2)
with col1:
    column_1 = st.selectbox('Колонка 1', (i for i in df.columns))

with col2:
    column_2 = st.selectbox('Колонка 2', (i for i in df.columns))
    categorial = st.checkbox('Является категориальной')

if column_1 == column_2:
    st.error('Нужно выбрать две РАЗНЫЕ колонки из датасета')
    exit(0)

if not categorial and ((df[column_2].dtypes == 'object') or (df[column_2].dtypes == 'string')):
    st.error('Скорее всего колонка 2 является категориальной')
    exit(0)


if categorial:
    st.markdown(f'Колличество данных')
    col1, col2 = st.columns(2)
    with col1:
        with st.spinner(f'Выводим...'):
            st.caption(f'Кол-во данных в выборке = {df[column_1].dropna().count()}')
            z = []
            x = {}
            z = set(df[column_1])
            for i in z:
                filter = f'{column_1} == "{i}"'
                x[i] = df.query(filter).dropna().count().to_numpy()[0]
                st.caption(f'Кол-во данных, где {column_1} = {i}: {x[i]}')     
    with col2:
        with st.spinner(f'Рисуем...'):
            for i in z:
                comment = '''
                fig = plt.figure(figsize=(5,1))
                a, b = [], []
                a.append(i)
                b.append(x[i])
                plt.barh(a, b)
                st.pyplot(fig)
                '''

    st.markdown(f'Распределения по ({column_1}, {column_2})')
    col1, col2 = st.columns(2)
    with col1:
        with st.spinner(f'Рисуем...'):
            fig = plt.figure(figsize=(5,5))
            df.groupby([column_1, column_2]).size().plot(kind='pie', subplots=True, autopct='%.1f', )
            st.pyplot(fig)
    with col2:
        with st.spinner(f'Рисуем...'):
            fig = plt.figure(figsize=(5,5))
            sns.histplot(data=df, x=column_1, y=column_2)
            st.pyplot(fig)
else:
    st.markdown(f'Колличество данных')
    with st.spinner(f'Рисуем...'):
        st.caption(f'Кол-во данных в датасете = {df[column_1].dropna().count()}')

    st.markdown(f'Распределение данных')

    col1, col2 = st.columns(2)
    with col1:
        with st.spinner(f'Рисуем...'):
            ax = df.boxplot(by=column_1, column=column_2, figsize=(5,5))
            ax.set_xlabel(column_1)
            ax.set_ylabel(column_2)
            st.pyplot(plt)

    with col2:
        with st.spinner(f'Рисуем...'):
            fig = plt.figure(figsize=(5,5))
            sns.histplot(data=df, x=column_1, y=column_2)
            st.pyplot(fig)

st.markdown(f'Статистический эксперимент')
st.caption(f'Определим две гипотезы:')
st.caption(f'''Для примера, в нашем датасете можно предположить, что нулевая гипотиза состоит в том, что значения рейтинга критиков и пользователей имеют взаимосвязь.
           Альтернативная гипотеза будет в том, что её нет.''')

st.markdown(f'Выбор методов расчёта при проверке гипотез')
if categorial:
    method = st.selectbox('Что используем?', (['chi-square test']))
else:
    method = st.selectbox('Что используем?', (['t-test', 'u-test']))

alpha = st.slider(f'Выбор минимального значения alpha при котором гипотиза подтверждается', 0.01, 0.2, 0.05)
st.caption(f'Значение alpha = {alpha}')

if method == 'chi-square test':
    data = pandas.crosstab(df[column_1], df[column_2])
    st.dataframe(data)
    res = scipy.stats.chi2_contingency(data)
    print(res)
    st.caption(f'p-value: {res.pvalue / 1:.8f}')

if method == 'u-test':
    z = set(df[column_1])
    for j, i in enumerate(z):
        filter = f'{column_1} == "{i}"'
        match j:
            case 0:
                group1 = df.query(filter)[column_2].dropna().to_numpy()
            case 1:
                group2 = df.query(filter)[column_2].dropna().to_numpy()


    res = scipy.stats.mannwhitneyu(group1, group2)
    print(res)
    st.caption(f'p-value: {res.pvalue / 1:.8f}')

if method == 't-test':
    z = set(df[column_1])
    for j, i in enumerate(z):
        filter = f'{column_1} == "{i}"'
        match j:
            case 0:
                group1 = df.query(filter)[column_2].dropna().to_numpy()
            case 1:
                group2 = df.query(filter)[column_2].dropna().to_numpy()


    res = scipy.stats.ttest_ind(group1, group2, equal_var=False)
    print(res)
    st.caption(f'p-value: {res.pvalue / 1:.8f}')

if res.pvalue > alpha:
    st.success('Можно предположить, что гипотеза подтвердилась')
else:
    st.error('Можно предположить, что гипотеза не подтвердилась')