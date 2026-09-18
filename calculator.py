import streamlit as st

st.title('Calculator App Using Streamlit')

num1=st.number_input('Enter the first number')
num2=st.number_input('Enter the second number')

st.write('**Operation**')

# st.text('select your operation')

operation=st.radio('Select your operation:',['Addition', 'Substraction', 'Multiplication', 'Division'])

if st.button('Calculate Result'):
    try:
        num1=float(num1)
        num2=float(num2)
        
        if operation=='Addition':
            result=num1+num2
            st.success(f'Answer = {result}')
        elif operation=='Substraction':
            result=num1-num2
            st.success(f'Answer = {result}')
        elif operation=='Multiplication':
            result=num1*num2
            st.success(f'Answer = {result}')
        elif operation=='Division':
            if num2==0:
                st.error('Division by zero')
            else:
                result=num1/num2
                st.success(f'Answer = {result}')
    except:
        st.error('Data yang diinput harus berupa angka')