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

      <form action="LoginController" method="post">
      <div class="col-lg-13">
        <div class="card mb-4">
          <div class="card-body">
         
         
            
            <div class="row">
              <div class="col-sm-3">
                <p class="mb-0" >First Name</p>
              </div>
              <div class="col-sm-9">
                <input type="text" name="firstname" class="text-muted mb-0" value="<%=rs.getString(2) %>" ></input>
              </div>
             
            </div>
             <hr>
            <div class="row">
              <div class="col-sm-3">
                <p class="mb-0" >Last Name</p>
              </div>
              <div class="col-sm-9">
                <input type="text" name="lastname" class="text-muted mb-0" value="<%=rs.getString(3) %>" ></input>
              </div>
             
            </div>
            <hr>
            
            <div class="row">
              <div class="col-sm-3">
                <p class="mb-0">Email</p>
              </div>
              <div class="col-sm-9">
                <input type="text" name="email" class="text-muted mb-0" value="<%=rs.getString(5) %>"></input>
              </div>
            </div>
            <hr>
            <div class="row">
              <div class="col-sm-3">
                <p class="mb-0">Phone</p>
              </div>
              <div class="col-sm-9">
                <input type="text" name="phone" class="text-muted mb-0" value="<%=rs.getString(6) %>"></input>
              </div>
            </div>
           
            <hr>
            <div class="row">
              <div class="col-sm-3">
                <p class="mb-0">Address</p>
              </div>
              <div class="col-sm-9">
                <input type="text" name="address"  class="text-muted mb-0" value="<%=rs.getString(7) %>"></input>
              </div>
            </div>
            <hr>
            <div class="row">
              <div class="col-sm-3">
                <p class="mb-0">Pincode</p>
              </div>
              <div class="col-sm-9">
                <input type="text" name="pincode" class="text-muted mb-0" value="<%=rs.getInt(9) %>"></input>
              </div>
            </div>
          </div>
          <input type="hidden" name="userid" value="<%=request.getParameter("customerid")%>">
        </div>
         </div>
<button name="action" type="submit" value="edit-profile" class="btn btn-primary" style="
    background-color: #ff8700;
    border-color: white;
    color: white;
    height: 43px
"><i class="fas fa-shopping-cart"></i> Update Profile</button>         
         </form>
         <%
                                         }
                                         cn.close();
				%>
    </div>
  
</section>
<jsp:include page="footer.jsp" />
</body>
</html>