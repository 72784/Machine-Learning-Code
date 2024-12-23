package com.controller;

import java.io.IOException;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import com.dao.HomeDao;
import com.dao.LoginDao;
import com.model.UserModel;
@WebServlet("/HomeController.do")
public class HomeController extends HttpServlet {
	private static final long serialVersionUID = 1L;
    public HomeController() {
        super();
        // TODO Auto-generated constructor stub
    }
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException 
	{
		String action=request.getParameter("action");
		if(action.equals("showuser"))
		{
			int userid=Integer.parseInt(request.getParameter("userid"));
			UserModel model=new HomeDao().verifyLogin(userid);
			request.setAttribute("model", model);
			request.getRequestDispatcher("home.jsp").forward(request, response);
		}
		
	}
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException 
	{
//		UserModel hmodel=new UserModel();
//		hmodel.setUserid(Integer.parseInt(request.getParameter("userid")));
//		hmodel.setFirstname(request.getParameter("firstname"));
//		hmodel.setLastname(request.getParameter("lastname"));
//		hmodel.setUsername(request.getParameter("username"));
//		hmodel.setPassword(request.getParameter("password"));
//		int x=new HomeDao().updateUser(hmodel);
//		if(x>0)
//		{
//			response.sendRedirect("HomeController.do?action=showuser&userid="+hmodel.getUserid());
//		}
//	}

}
}