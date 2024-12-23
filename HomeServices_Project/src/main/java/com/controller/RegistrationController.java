 package com.controller;

import java.io.IOException;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.dao.RegistrationDao;
import com.model.UserModel;

@WebServlet("/RegistrationController")
public class RegistrationController extends HttpServlet {
	private static final long serialVersionUID = 1L;
       

    public RegistrationController() {
        super();
        // TODO Auto-generated constructor stub
    }


	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		
	}

	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {

			UserModel rmodel=new UserModel();
			
			//here, name is written exactly the name you wrote in form page in signup.jsp
			rmodel.setFirstname(request.getParameter("firstname"));
			rmodel.setLastname(request.getParameter("lastname"));
			rmodel.setGender(request.getParameter("genderradio"));
			rmodel.setEmailaddress(request.getParameter("emailaddress"));
			rmodel.setPhonenumber(request.getParameter("phonenumber"));
			rmodel.setAddress(request.getParameter("address"));
			rmodel.setCity(request.getParameter("city"));
			rmodel.setPincode(Integer.parseInt(request.getParameter("pincode"))) ;
			rmodel.setPassword(request.getParameter("password"));
			
			int x=new RegistrationDao().userRegistration(rmodel);
			if(x>0)
			{
				request.setAttribute("msg", "Record inserted...");
				request.getRequestDispatcher("login.jsp").forward(request, response);
			}
			else
			{
				request.setAttribute("msg", "Record not inserted...");
				request.getRequestDispatcher("signup.jsp").forward(request, response);
			}		
		
	}

}
