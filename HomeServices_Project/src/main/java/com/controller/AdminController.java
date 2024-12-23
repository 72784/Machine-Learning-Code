package com.controller;

import java.io.File;
import java.io.IOException;
import java.util.Properties;
import java.util.Random;

import javax.mail.Authenticator;
import javax.mail.Message;
import javax.mail.PasswordAuthentication;
import javax.mail.Session;
import javax.mail.Transport;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeMessage;
import javax.servlet.ServletException;
import javax.servlet.annotation.MultipartConfig;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import javax.servlet.http.Part;
import javax.swing.Action;

import com.dao.AdminDao;
import com.dao.ForgotDao;
import com.dao.HomeDao;
import com.dao.RegistrationDao;
import com.model.AssignModel;
import com.model.ServiceModel;
import com.model.SubServiceModel;
import com.model.UserModel;
import com.model.AssignModel;
import com.model.BookModel;
import com.model.EmployeeModel;

@WebServlet("/AdminController")
@MultipartConfig(fileSizeThreshold =1024*1024*512,//512MB
maxFileSize = 1024*1024*512,//512MB
maxRequestSize = 1024*1024*512) //512MB
public class AdminController extends HttpServlet {
	private static final long serialVersionUID = 1L;
       
    
    public AdminController() {
        super();
    }
private String extractfilename(Part file) {
	String cd= file.getHeader("content-disposition");
	System.out.println(cd);
	String[] items = cd.split(";");
	for(String string : items) {
		if(string.trim().startsWith("filename")) {
			return string.substring(string.indexOf("=")+2,string.length()-1);
		}
	}
	return "";
}
	protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
			
	}

	
	protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
		String action=request.getParameter("action");
		if(action.equalsIgnoreCase("page-login")) {
			String username=request.getParameter("username");
			String password=request.getParameter("password");
			if(username.equals("admin")&&password.equals("admin")) 
			{
			response.sendRedirect("admin/index.jsp");	
			}
			else
			{
				response.sendRedirect("admin/page-login.jsp");	
	
			}
		}
		else if(action.equalsIgnoreCase("main-service"))
		{
			
			String savePath = "C:\\Users\\msved\\Downloads\\Website (1)\\Website\\WebProject\\src\\main\\webapp\\uploadimage";
			
			File filesaveDir = new File(savePath);
			if(!filesaveDir.exists()) {
				filesaveDir.mkdir();
			}
			Part file= request.getPart("image");
			System.out.println(file);
			String fileName=extractfilename(file);
			System.out.println("first:"+fileName);
					
			file.write(savePath+File.separator+fileName);
			System.out.println("second:"+fileName);
			String filePath=savePath+File.separator+fileName;
		
			System.out.println("Your image path is..."+filePath);;
			
		ServiceModel amodel=new ServiceModel();
//		amodel.setServiceid(Integer.parseInt(request.getParameter("serviceid"))) ;
amodel.setType(request.getParameter("type"));
amodel.setDescription(request.getParameter("description"));
amodel.setImage(fileName);
int x=new AdminDao().addData(amodel);
if(x>0)
{
	request.setAttribute("msg", "Record inserted...");
	response.sendRedirect("admin/index.jsp");
//	request.getRequestDispatcher("admin/index.jsp").forward(request, response);
}
else
{
	request.setAttribute("msg", "Record not inserted...");
	request.getRequestDispatcher("main-service.jsp").forward(request, response);
}
		}
		else if(action.equalsIgnoreCase("sub-service"))
		{
			SubServiceModel ssm=new SubServiceModel();
			ssm.setParentid(Integer.parseInt(request.getParameter("serviceid")));
			ssm.setTitle(request.getParameter("title"));
			ssm.setDescription(request.getParameter("description"));
			ssm.setCost(request.getParameter("cost"));
		
			int x=new AdminDao().addSubData(ssm);
		if(x>0) {
			response.sendRedirect("admin/index.jsp");
		}
		else
		{
			response.sendRedirect("admin/sub-service.jsp");

		}
		}else if(action.equalsIgnoreCase("employee"))
		{
			EmployeeModel ep=new EmployeeModel();
//			ep.setEmployee_id(Integer.parseInt(request.getParameter("employee_id"))); do not set primary key ever
			ep.setFirstname(request.getParameter("firstname"));
			ep.setLastname(request.getParameter("lastname"));
			ep.setEmail(request.getParameter("email"));
			ep.setPhone(request.getParameter("phone"));
			ep.setPassword(request.getParameter("password"));
			ep.setSpecialization(request.getParameter("specialization"));
			ep.setGender(request.getParameter("gender"));
			
			int x=new AdminDao().addEmployeeData(ep);
		if(x>0) {
			response.sendRedirect("RoyalUI-Admin-master\\login.jsp");
		}
		else
		{
			response.sendRedirect("RoyalUI-Admin-master/service_men_sign_up.jsp");

		}
		}
		else if(action.equalsIgnoreCase("contract"))
		{
			BookModel bmodel=new BookModel();
			bmodel.setBookid(Integer.parseInt(request.getParameter("bookid")));
			AssignModel amodel = new AssignModel();
			amodel.setBookid(Integer.parseInt(request.getParameter("bookid")));
			amodel.setDate(request.getParameter("date"));
			amodel.setTime(request.getParameter("time"));
			int x=new AdminDao().updateUser(bmodel);
			if(x>0)
			{
				int y = new AdminDao().insertdatetime(amodel);
				if(y>0) {
//					String email=request.getParameter("email");
					
//					ForgotDao dao=new ForgotDao();
					
						String emailid=request.getParameter("email");
						String username=request.getParameter("servicemen");
String time=request.getParameter("time");
String date=request.getParameter("date");
						final String Senderid = "nearbyservices2023@gmail.com";
						final String password = "hiljxkjbxuaipair";

						Properties props = new Properties();

						String host = "smtp.gmail.com";

						
						props.put("mail.smtp.socketFactory.port", "465");
						props.put("mail.smtp.socketFactory.class", "javax.net.ssl.SSLSocketFactory");
						props.put("mail.smtp.socketFactory.fallback", "true");
						props.put("mail.smtp.ssl.protocols", "TLSv1.2");
						props.put("mail.smtp.auth", "true");
						props.put("mail.smtp.debug", "true");
						props.put("mail.smtp.starttls.enable", "true");
						props.put("mail.smtp.host", host);
						props.put("mail.smtp.port", "465");
						props.put("mail.smtp.user", Senderid);
						props.put("mail.smtp.password", password);

						Session session = Session.getInstance(props, new Authenticator() {
							protected PasswordAuthentication getPasswordAuthentication() {
								return new PasswordAuthentication(Senderid, password);
							}
						});


						try {
							Random rand = new Random();
							int otp = rand.nextInt(900000) + 100000;
							Message message = new MimeMessage(session);
							message.setFrom(new InternetAddress(Senderid));
							message.setRecipients(Message.RecipientType.TO, InternetAddress.parse(emailid));// to
																											// mail
																											// address.
							message.setSubject("OTP Request");
							String msg1 = "<!DOCTYPE html><html><head></head><body><div style='background-color:#f8f8f8; width:500px; height:200px'><div style='background-color:#00e58b; width:500px; height:50px'><h3 style='color:white; padding-top:10px;'> Service Confirmation Email </h3></div><p style='color:gray; margin-left:-300px;'>Dear "
									+ username
									+ ",</p><p style='color:gray;'>Your booking is confirmed,<br> Our servicmen will arrive on "
									+ request.getParameter("date") + " between " + request.getParameter("time") + "<br><br>"
									+ otp
									+ "  is your One Time Password.<br>Verify this with our servicemen before obtaining the benefits of our services.</p><p>After the completion of service you can fill the feedback form available on our website to help other customers.</p><div><p></body></html>";
							message.setContent(msg1, "text/html; charset=utf-8");
					
							Transport.send(message);

							HttpSession otpsession = request.getSession();
							otpsession.setAttribute("otp", otp);
//							otpsession.setMaxInactiveInterval(10 * 60); /*Session Set for 10 minutes*/
//							otpsession.setAttribute("UserData", f);
							response.sendRedirect("RoyalUI-Admin-master\\login.jsp");

						} catch (Exception e) {
							request.setAttribute("msg", "Otp Not Send");
							request.getRequestDispatcher("RoyalUI-Admin-master\\contract_details.jsp").forward(request, response);
							e.printStackTrace();
						}
					}
					
					
				
				else {
					System.out.println("Insertion of date time failed");
				}
//				response.sendRedirect("RoyalUI-Admin-master\\index.jsp");
			}	
			else {
				System.out.println("Inside else");
			}
}else if(action.equalsIgnoreCase("request"))
		{
			AssignModel asmodel=new AssignModel();
			asmodel.setBookid(Integer.parseInt(request.getParameter("bookid")));
//			asmodel.setAssign_id(Integer.parseInt(request.getParameter("assign_id")));do not set primary key ever
			asmodel.setServicemen(request.getParameter("servicemen"));

			int x=new AdminDao().addAssignData(asmodel);
			if(x>0)
			{
				response.sendRedirect("admin\\booking.jsp");
			}	
			else {
				System.out.println("Inside else");
			}
		}else if (action.equalsIgnoreCase("requestdenied"))
		{
			BookModel bmodel = new BookModel();
			AssignModel amodel = new AssignModel();
			
			bmodel.setBookid(Integer.parseInt(request.getParameter("bookid")));
			
			
			amodel.setBookid(Integer.parseInt(request.getParameter("bookid")));
			
			System.out.println("before status update");
			
			int x = new AdminDao().requestdenied(bmodel);
			if (x>0){
				int y = new AdminDao().cancelassignment(amodel);
				if (y>0){
					response.sendRedirect("RoyalUI-Admin-master\\login.jsp");
				}
				else {
					System.out.println("problem in update assignment");
					System.out.println(amodel.getServicemen());
				}
				
			}
			else {
			}
		}else if(action.equalsIgnoreCase("completed"))
		{
			BookModel ep=new BookModel();
			ep.setBookid(Integer.parseInt(request.getParameter("bookid")));

//			ep.setEmployee_id(Integer.parseInt(request.getParameter("employee_id"))); do not set primary key ever
			ep.setExtracost(request.getParameter("extra_cost"));
			
			int x=new AdminDao().addExtraCostData(ep);
		if(x>0) {
			response.sendRedirect("RoyalUI-Admin-master\\login.jsp");
		}
		else
		{
			response.sendRedirect("RoyalUI-Admin-master/service_men_sign_up.jsp");

		}

}else if (action.equalsIgnoreCase("cancel-request-by-admin"))
{
	BookModel bmodel = new BookModel();
	AssignModel amodel = new AssignModel();
	
	bmodel.setBookid(Integer.parseInt(request.getParameter("bookid")));
	
	
	amodel.setBookid(Integer.parseInt(request.getParameter("bookid")));
	
	System.out.println("before status update");
	
	int x = new AdminDao().cancelrequest(bmodel);
	if (x>0){
		int y = new AdminDao().cancelassignment(amodel);
		if (y>0){
			response.sendRedirect("admin/booking.jsp");
		}
		else {
			System.out.println("problem in update assignment");
			System.out.println(amodel.getServicemen());
		}
		
	}
	else {
	}
}
	}
}
