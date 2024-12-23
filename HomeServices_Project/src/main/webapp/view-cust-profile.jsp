<%@page import="java.io.Console"%>
<%@page import="java.sql.ResultSet"%>
<%@page import="java.sql.PreparedStatement"%>
<%@page import="com.util.DBUtil"%>
<%@page import="java.sql.Connection"%>
<%@ page language="java" contentType="text/html; charset=ISO-8859-1"
    pageEncoding="ISO-8859-1"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="ISO-8859-1">
<title>Insert title here</title>
</head>
<body style="background-color: #eee;">
<jsp:include page="styles.jsp" />


<jsp:include page="cust-header.jsp" />

<h3 style="margin-top:126px;text-align: center;margin-bottom:0px">My Profile</h3>
<section style="background-color: #eee;margin-top:30px;margin-bottom:54px">
  <div class="container py-5" style="margin-top:-47px">
   
 <%
     Connection cn=new DBUtil().getDatabaseConnection();
 	
     PreparedStatement st = cn.prepareStatement("select * from customer where userid=?");
     st.setInt(1, Integer.parseInt(request.getParameter("customerid")));
     ResultSet rs=st.executeQuery();
     while(rs.next())
     {
%>
    <div class="row">
      <div class="col-lg-4">
        <div class="card mb-4">
          <div class="card-body text-center" style="height:366px;margin-top:20px">
               <%
									if (rs.getString("gender").equalsIgnoreCase("Male")) {
									%>       
            <img src="male_logo_2.png" alt="avatar"
              class="rounded-circle img-fluid" style="width: 150px;height:150px">
              <%}else{ %>
               <img src="female_logo.png" alt="avatar"
              class="rounded-circle img-fluid" style="width: 150px;height:150px">
           
              <%} %>
              <div style="padding-top: 10px;padding-bottom: 15px;">
            <h5 class="my-3" ><%=rs.getString(2)%></h5>
           
            </div>
            <div class="d-flex justify-content-center mb-2">
  
                        <button type="submit" style="margin: auto; display:block; width: 150px"   class ="btn btn-outline-primary ms-1"> <a href="view-cust-edit-profile.jsp?customerid=<%=request.getParameter("customerid")%>"> Edit Profile </a></button>
      
            </div>
                         <a href="login.jsp"> <button type="button" style="width:150px" class="btn btn-outline-danger ms-1">Sign Out</button></a>
            
          </div>
        </div>
        
      </div>


      
      <div class="col-lg-8">
        <div class="card mb-4">
          <div class="card-body">
          <div class="row">
              <div class="col-sm-3">
                <p class="mb-0">User Id</p>
              </div>
              <div class="col-sm-9">
                <p class="text-muted mb-0" ><%=rs.getInt(1) %></p>
              </div>
            </div>
         
            <hr>
            <div class="row">
              <div class="col-sm-3">
                <p class="mb-0" >Full Name</p>
              </div>
              <div class="col-sm-9">
                <p class="text-muted mb-0" ><%=rs.getString(2) + " "+rs.getString(3)%></p>
              </div>
             
            </div>
               
             <hr>
            <div class="row">
              <div class="col-sm-3">
                <p class="mb-0">Gender</p>
              </div>
              <div class="col-sm-9">
                <p class="text-muted mb-0"><%=rs.getString(4) %></p>
              </div>
            </div>
            <hr>
            
            <div class="row">
              <div class="col-sm-3">
                <p class="mb-0">Email</p>
              </div>
              <div class="col-sm-9">
                <p class="text-muted mb-0"><%=rs.getString(5) %></p>
              </div>
            </div>
            <hr>
            <div class="row">
              <div class="col-sm-3">
                <p class="mb-0">Phone</p>
              </div>
              <div class="col-sm-9">
                <p class="text-muted mb-0"><%=rs.getString(6) %></p>
              </div>
            </div>
           
            <hr>
            <div class="row">
              <div class="col-sm-3">
                <p class="mb-0">Address</p>
              </div>
              <div class="col-sm-9">
                <p class="text-muted mb-0"><%=rs.getString(7) %></p>
              </div>
            </div>
            <hr>
            <div class="row">
              <div class="col-sm-3">
                <p class="mb-0">Pincode</p>
              </div>
              <div class="col-sm-9">
                <p class="text-muted mb-0"><%=rs.getInt(9) %></p>
              </div>
            </div>
          </div>
          
        </div>
         </div>
         <%
                                         }
                                         cn.close();
				%>
    </div>
  </div>
</section>
<jsp:include page="footer.jsp" />
</body>
</html>