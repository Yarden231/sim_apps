import streamlit as st

def show_theory_lab():
    st.header("Classes Theory Lab")

    st.write("""
    ## Introduction to Classes

    Classes are a fundamental concept in object-oriented programming (OOP). They allow you to create custom data types 
    that encapsulate data and behavior. In this lab, we'll explore the basics of classes and how they can be used in 
    the context of our food truck simulation.

    ### Basic Class Structure

    Here's a basic example of a class:
    """)

    st.code("""
    class Rocket:
        def __init__(self, x=0, y=0):
            self.x = x
            self.y = y

        def move_up(self):
            self.y += 1
    """, language="python")

    st.write("""
    This `Rocket` class has:
    - An `__init__` method (constructor) that initializes the rocket's position
    - A `move_up` method that increases the y-coordinate

    ### Creating Objects

    You can create objects (instances) of a class like this:
    """)

    st.code("""
    my_rocket = Rocket(10, 20)
    print(f"Rocket position: ({my_rocket.x}, {my_rocket.y})")
    my_rocket.move_up()
    print(f"New position: ({my_rocket.x}, {my_rocket.y})")
    """, language="python")

    st.write("""
    ### Inheritance

    Classes can inherit properties and methods from other classes. This allows for code reuse and the creation of 
    specialized versions of classes.
    """)

    st.code("""
    class Shuttle(Rocket):
        def __init__(self, x=0, y=0, flights=0):
            super().__init__(x, y)
            self.flights = flights

        def complete_mission(self):
            self.flights += 1
    """, language="python")

    st.write("""
    In this example, `Shuttle` inherits from `Rocket` and adds a new attribute `flights` and a new method `complete_mission`.

    ### Application to Food Truck Simulation

    In our food truck simulation, we use classes to model various components:

    1. `FoodTruck`: Represents the food truck itself
    2. `Customer`: Represents a customer in the simulation
    3. `Order`: Represents a customer's order

    These classes help organize our code and make it easier to manage the complex behavior of the simulation.

    ### Exercise

    Try creating a simple `FoodTruck` class with the following features:
    - Attributes for the truck's name and menu items
    - A method to add items to the menu
    - A method to display the current menu

    Use the code editor below to implement this class:
    """)

    code = st.text_area("Code Editor", 
    """
    class FoodTruck:
        def __init__(self, name):
            self.name = name
            self.menu = {}

        def add_item(self, item, price):
            # Your code here

        def display_menu(self):
            # Your code here

    # Test your class
    my_truck = FoodTruck("Tasty Bites")
    my_truck.add_item("Burger", 5.99)
    my_truck.add_item("Fries", 2.99)
    my_truck.display_menu()
    """, height=300)

    if st.button("Run Code"):
        try:
            exec(code)
            st.success("Code executed successfully!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    st.write("""
    This lab provides a basic introduction to classes and their application in our food truck simulation. 
    As you work through the simulation, you'll see how these concepts are applied in a more complex scenario.
    """)

if __name__ == "__main__":
    show_theory_lab()
