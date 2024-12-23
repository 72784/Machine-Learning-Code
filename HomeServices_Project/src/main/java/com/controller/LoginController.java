package com.controller;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import com.dao.LoginDao;
import com.model.AssignModel;
import com.model.BookModel;
import com.model.FeedbackModel;
import com.model.EmployeeModel;
import com.model.UserModel;
@WebServlet("/LoginController")
public class LoginController extends HttpServlet {
	private static final long serialVersionUID = 1L;
       
    public LoginController() {
        super();
    }
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException 
	{
		
	}
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException 
	{
		String action=request.getParameter("action");
		if(action.equalsIgnoreCase("login"))
		{
		UserModel lmodel=new UserModel();
		lmodel.setEmailaddress(request.getParameter("emailaddress"));
		lmodel.setPassword(request.getParameter("password"));
		
		UserModel model=new LoginDao().getLogin(lmodel);
		if(model != null)
		{
			System.out.println("inside if");
			HttpSession session=request.getSession(true);
			session.setAttribute("model", model);
			session.setMaxInactiveInterval(60);//Session Timeout of 60s
			response.sendRedirect("afterLogin.jsp");
		}
		else  
		{
			System.out.println("inside else");
			request.setAttribute("msg", "Invalid Username or password...");
			request.getRequestDispatcher("login.jsp").forward(request, response);
		}
		}else if(action.equalsIgnoreCase("book order"))
		{
			BookModel bmodel=new BookModel();
			
			bmodel.setId(Integer.parseInt(request.getParameter("id")));
			bmodel.setUserid(Integer.parseInt(request.getParameter("userid")));
			bmodel.setPrice(request.getParameter("price"));
			bmodel.setStatus("Pending");
			bmodel.setPstatus("Successful");
			
			Date d1 = new Date();
			SimpleDateFormat s1 = new SimpleDateFormat("dd-MM-yyyy");
			bmodel.setBdate(s1.format(d1));
			
			int x=new LoginDao().customerBooking(bmodel);
			if(x>0)
			{
				response.sendRedirect("payment.jsp?price="+bmodel.getPrice());
			}
		}else if(action.equalsIgnoreCase("employee-log-in"))
		{
			
			EmployeeModel ep=new EmployeeModel();
			ep.setEmail(request.getParameter("email"));
			ep.setPassword(request.getParameter("password"));
			
			
			EmployeeModel model=new LoginDao().getEmployeeLogin(ep);
			if(model != null)
			{
				System.out.println("inside if");
				HttpSession session=request.getSession(true);
				session.setAttribute("model", model);

				session.setMaxInactiveInterval(60);//Session Timeout of 60s
				response.sendRedirect("RoyalUI-Admin-master\\index.jsp?servicemen="+model.getFirstname()+" "+model.getLastname());

			}
			else  
			{
				System.out.println("inside else");
				request.setAttribute("msg", "Invalid Username or password...");
				request.getRequestDispatcher("login.jsp").forward(request, response);
			}
		}else if(action.equalsIgnoreCase("feedback")) {
FeedbackModel fmodel=new FeedbackModel();
			
			fmodel.setUserid(Integer.parseInt(request.getParameter("userid")));
			fmodel.setFullname(request.getParameter("fullname"));
			fmodel.setSubject(request.getParameter("subject"));
			fmodel.setComment(request.getParameter("comment"));
			fmodel.setEmail(request.getParameter("email"));
			
			
			
			int x=new LoginDao().feedback(fmodel);
			if(x>0)
			{
				
				response.sendRedirect("myorders.jsp?customerid="+fmodel.getUserid());
			}
		}else if(action.equalsIgnoreCase("feedback1")) {
FeedbackModel fmodel=new FeedbackModel();
			
//			fmodel.setUserid(Integer.parseInt(request.getParameter("userid")));
			fmodel.setFullname(request.getParameter("fullname"));
			fmodel.setSubject(request.getParameter("subject"));
			fmodel.setComment(request.getParameter("comment"));
			fmodel.setEmail(request.getParameter("email"));
			
			
			
			int x=new LoginDao().feedback1(fmodel);
			if(x>0)
			{
				
				response.sendRedirect("login.jsp");
			}
		}else if(action.equalsIgnoreCase("edit-profile")) {
UserModel fmodel=new UserModel();
			
			fmodel.setFirstname(request.getParameter("firstname"));
			fmodel.setLastname(request.getParameter("lastname"));
			fmodel.setEmailaddress(request.getParameter("email"));
			fmodel.setPhonenumber(request.getParameter("phone"));
			fmodel.setAddress(request.getParameter("address"));
			fmodel.setPincode(Integer.parseInt(request.getParameter("pincode")));
			fmodel.setCustomerid(Integer.parseInt(request.getParameter("userid")));
			
			
			
			int x=new LoginDao().update_profile(fmodel);
			if(x>0)
			{
				
				response.sendRedirect("view-cust-profile.jsp?customerid="+fmodel.getCustomerid());
			}
		}
	}

}
