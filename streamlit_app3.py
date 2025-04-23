import streamlit as st
import matplotlib.pyplot as plt
from expn_infer import resolve_symbols_on_img, predict_latex_expression
import re
import os
from PIL import Image
import io
import sympy
from sympy import symbols, sympify, solve, Eq, simplify, N, E, Symbol, Function, dsolve, Derivative

def latex_to_sympy(latex_expr):
    """Convert LaTeX math expression to SymPy expression with proper derivative handling"""
    # Check for derivative notation in LaTeX format: \frac{d}{dx}(...)
    diff_match = re.search(r'\\frac{\s*d\s*}{\s*d\s*([a-zA-Z]+)\s*}\s*\(\s*(.*?)\s*\)', latex_expr)
    if diff_match:
        var = diff_match.group(1)
        expr = diff_match.group(2)
        
        # Pre-process the expression for sympify
        expr = expr.replace("^{", "**").replace("}", "")
        expr = expr.replace(" ", "")
        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
        
        
        # Return a SymPy Derivative representation
        return f"Derivative({expr}, {var})"
    
    # Check for integral notation in LaTeX format: \int (expression) dx
    int_match = re.search(r'\\int\s*\(\s*(.*?)\s*\)\s*d\s*([a-zA-Z]+)', latex_expr)
    if int_match:
        expr = int_match.group(1)
        var = int_match.group(2)
        
        # Pre-process the expression for sympify
        expr = expr.replace("^{", "**").replace("}", "")
        # Remove spaces before applying regex
        expr = expr.replace(" ", "")
        # Add multiplication symbol between numbers and variables
        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
        
        # Return a SymPy Integral representation
        return f"Integral({expr}, {var})"
    
    # If not a derivative, proceed with normal conversions
    # Replace LaTeX fraction with sympy division
    latex_expr = re.sub(r'\\frac{(.*?)}{(.*?)}', r'(\1)/(\2)', latex_expr)
    
    # Replace LaTeX exponents with sympy exponents
    latex_expr = latex_expr.replace("^{", "**").replace("}", "")
    
    # Remove spaces
    latex_expr = latex_expr.replace(" ", "")
    
    # Add multiplication symbol between numbers and variables
    latex_expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', latex_expr)
    
    # Replace 'e' with E (sympy's representation of e)
    latex_expr = re.sub(r'(?<![a-zA-Z])e(?![a-zA-Z])', 'E', latex_expr)

    # Replace '\pi' with pi (sympy's representation of pi)
    latex_expr = latex_expr.replace(r'\pi', ' *pi')
    
    return latex_expr

def solve_with_streamlit(latex_expr):
    """Modified version of solve_or_evaluate for Streamlit with derivative and integral support"""
    normalized_expr = latex_to_sympy(latex_expr)
    st.text(f"Normalized Expression: {normalized_expr}")
    
    # Check if the expression is a derivative
    if normalized_expr.startswith("Derivative("):
        try:
            # Extract the expression and variable from the derivative notation
            match = re.match(r"Derivative\((.*),\s*([a-zA-Z]+)\)", normalized_expr)
            if match:
                expr_str, var_str = match.groups()
                
                # Create symbolic expression and variable
                expr = sympify(expr_str)
                var = Symbol(var_str)
                
                # Calculate the derivative
                derivative = sympy.diff(expr, var)
                
                st.success("Derivative computed:")
                st.latex(sympy.latex(derivative))
                
                # Simplify the result
                simplified = simplify(derivative)
                if simplified != derivative:
                    st.text("Simplified form:")
                    st.latex(sympy.latex(simplified))
                
                return
        except Exception as e:
            st.error(f"Error computing derivative: {str(e)}")
    
    # Check if the expression is an integral
    if normalized_expr.startswith("Integral("):
        try:
            # Extract the expression and variable from the integral notation
            match = re.match(r"Integral\((.*),\s*([a-zA-Z]+)\)", normalized_expr)
            if match:
                expr_str, var_str = match.groups()
                
                # Create symbolic expression and variable
                expr = sympify(expr_str)
                var = Symbol(var_str)
                
                # Calculate the integral
                integral = sympy.integrate(expr, var)
                
                st.success("Indefinite integral computed:")
                st.latex(sympy.latex(integral) + " + c")
                
                
                return
        except Exception as e:
            st.error(f"Error computing integral: {str(e)}")
    
    # Continue with original functionality for non-derivatives and non-integrals
    if "=" in normalized_expr:
        lhs, rhs = normalized_expr.split("=")
        lhs = lhs.strip()
        rhs = rhs.strip()
        
        # Create the equation
        try:
            equation = Eq(sympify(lhs), sympify(rhs))
            st.text(f"SymPy Expression: {equation}")

            # Identify variables in the equation
            variables = list(set(sympy.Symbol(sym) for sym in re.findall(r'[a-zA-Z]+', normalized_expr) if sym not in ['E', 'pi']))

            if not variables:
                st.warning("No variables found in the equation.")
                return

            # If there's only one variable, solve for it
            if len(variables) == 1:
                solutions = solve(equation, variables[0])
                numerical_solutions = [N(sol, 10) for sol in solutions]
                st.success(f"Solutions for {variables[0]}: {numerical_solutions}")
            else:
                st.info(f"Multiple variables found: {variables}")
                # Create a dropdown for variable selection
                var_to_solve = st.selectbox(
                    "Select variable to solve for:",
                    [str(var) for var in variables]
                )
                var_symbol = sympy.Symbol(var_to_solve)
                if var_symbol in variables:
                    solutions = solve(equation, var_symbol)
                    numerical_solutions = [N(sol, 10) for sol in solutions]
                    st.success(f"Solutions for {var_symbol}: {numerical_solutions}")
                else:
                    st.error(f"Variable {var_to_solve} not found in the equation.")

            # Simplify the equation
            simplified_eq = simplify(equation)
            st.text(f"Simplified equation: {simplified_eq}")
        except Exception as e:
            st.error(f"Error solving equation: {str(e)}")

    else:
        # If it's just an expression, evaluate it
        try:
            expr = sympify(normalized_expr)
            simplified_expr = simplify(expr)
            st.text(f"Simplified expression: {simplified_expr}")
            
            # Try to evaluate numerically
            try:
                numerical_value = N(simplified_expr, 10)
                st.success(f"Numerical value: {numerical_value}")
            except:
                st.warning("Unable to evaluate numerically. The expression may contain variables.")
        except Exception as e:
            st.error(f"Error evaluating expression: {str(e)}")


def main():
    st.title("Handwritten Mathematical Equation Solver")
    st.write("Upload an image of a handwritten mathematical equation to get started!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Create a temporary directory if it doesn't exist
        os.makedirs('temp', exist_ok=True)
        
        # Save the uploaded file temporarily
        temp_path = os.path.join('temp', "temp_image.jpg")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            # Process the image
            symbols, _, stacked_list, script_levels, _, ax = resolve_symbols_on_img(temp_path)

            fig = ax.figure
            # fig.set_size_inches(4,3)
            
            # Display the processed image with bounding boxes
            st.write("### Processed Image with Symbol Detection")
            # fig = ax.figure
            st.pyplot(fig)
            plt.close(fig)

            # Make prediction
            model_path = 'final_math_symbol_cnn.pth'
            latex_expr = predict_latex_expression(symbols, script_levels, stacked_list, model_path)
            latex_expr = re.sub(r"\\frac(?!{)", "-", latex_expr)

            # Display the predicted LaTeX
            st.write("### Predicted LaTeX Expression")
            st.code(latex_expr, language="latex")

            # Allow user to edit the LaTeX
            st.write("### Edit LaTeX (if needed)")
            corrected_latex = st.text_input("Edit LaTeX expression", value=latex_expr)

            # Display the rendered equation
            st.write("### Rendered Equation")
            st.latex(corrected_latex)

            # Add solve/simplify option
            if st.button("Solve/Simplify Expression"):
                st.write("### Solution")
                try:
                    with st.spinner("Solving..."):
                        solve_with_streamlit(corrected_latex)
                except Exception as e:
                    st.error(f"Error solving equation: {str(e)}")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    st.set_page_config(page_title="Math Equation Solver", layout="wide")
    main()