<!DOCTYPE html>
<%@page import="java.text.SimpleDateFormat"%>
<%@page import="java.util.Date"%>
<%@page import="java.sql.ResultSet"%>
<%@page import="java.sql.PreparedStatement"%>
<%@page import="com.util.DBUtil"%>
<%@page import="java.sql.Connection"%>
<html lang="en">

<head>
<!-- Required meta tags -->
<meta charset="utf-8">
<meta name="viewport"
	content="width=device-width, initial-scale=1, shrink-to-fit=no">
<title>RoyalUI Admin</title>
<!-- plugins:css -->
<link rel="stylesheet" href="vendors/ti-icons/css/themify-icons.css">
<link rel="stylesheet" href="vendors/base/vendor.bundle.base.css">
<!-- endinject -->
<!-- plugin css for this page -->
<!-- End plugin css for this page -->
<!-- inject:css -->
<link rel="stylesheet" href="css/style.css">
<!-- endinject -->
<link rel="shortcut icon" href="images/favicon.png" />
</head>
<body>
<jsp:include page="headers.jsp" />
<%
										Connection cn2 = new DBUtil().getDatabaseConnection();
										PreparedStatement stsecond = cn2.prepareStatement("select * from book where bookid=?");
										stsecond.setString(1, request.getParameter("bookid"));
										ResultSet rs2 = stsecond.executeQuery();

										while (rs2.next()) {
										%>
										<%
										Connection cn3 = new DBUtil().getDatabaseConnection();
										PreparedStatement stthird = cn3.prepareStatement("select * from customer where userid=?");
										stthird.setString(1, rs2.getString("userid"));
										ResultSet rs3 = stthird.executeQuery();

										while (rs3.next()) {
										%>
	

			<div class="main-panel">
				<div class="content-wrapper" style="background-color: white">
					
						<h4 class="card-title display-4">Contract Details</h4>
						<div class="card-body">
                  
                  <form class="forms-sample" method="post" action="../AdminController">
                    <div class="form-group row" style="margin-bottom:0px">
                      <label for="exampleInputUsername2" class="col-sm-3 col-form-label">Name</label>
                      <div class="col-sm-9">
                        <input readonly name="servicemen"value="<%=rs3.getString("firstname") + " " + rs3.getString("lastname")%>
										" type="text" class="form-control" id="exampleInputUsername2" >
                      </div>
                    </div>
                    <div class="form-group row" style="margin-bottom:0px">
                      <label for="exampleInputMobile" class="col-sm-3 col-form-label">Service Requirement</label>
                      <div class="col-sm-9">
                       <%
                                             Connection cn1=new DBUtil().getDatabaseConnection();
                                             PreparedStatement stfirst = cn1.prepareStatement("select * from book where bookid=?");
                    		  stfirst.setInt(1, Integer.parseInt(request.getParameter("bookid")));
                                         ResultSet rs1=stfirst.executeQuery();
                                         while(rs1.next())
                                         {
%>
<%
                                             Connection cn4=new DBUtil().getDatabaseConnection();
                                             PreparedStatement stfourth = cn4.prepareStatement("select * from subservicei where id=?");
                    		  stfourth.setString(1, rs1.getString("id"));
                                         ResultSet rs4=stfourth.executeQuery();
                                         while(rs4.next())
                                         {
%>
                        <input  value="<%=rs4.getString("title")%>" readonly  class="form-control" id="exampleInputMobile">
                      <%}cn4.close(); %>
                      <%}cn1.close(); %>
                      </div>
                    </div>
                    <div class="form-group row" style="margin-bottom:0px">
                      <label for="exampleInputEmail2" class="col-sm-3 col-form-label">Email</label>
                      <div class="col-sm-9">
                        <input type="email" name="email" readonly value="<%=rs3.getString("emailaddress")%>
										"  class="form-control" id="exampleInputEmail2">
                      </div>
                    </div>
                    <div class="form-group row" style="margin-bottom:0px">
                      <label for="exampleInputMobile" class="col-sm-3 col-form-label">Mobile</label>
                      <div class="col-sm-9">
                        <input  value="<%=rs3.getString("phonenumber")%>" readonly  class="form-control" id="exampleInputMobile" ">
                      </div>
                    </div>
                    
                    <div class="form-group row" style="margin-bottom:0px">
                      <label for="exampleInputMobile" class="col-sm-3 col-form-label">Address</label>
                      <div class="col-sm-9">
                        <input  value="<%=rs3.getString("address")%>" readonly  class="form-control" id="exampleInputMobile">
                      </div>
                    </div>
                    <div class="form-group row" style="margin-bottom:0px">
                      <label for="exampleInputMobile" class="col-sm-3 col-form-label">City</label>
                      <div class="col-sm-9">
                        <input  value="<%=rs3.getString("city")%>" readonly  class="form-control" id="exampleInputMobile">
                      </div>
                    </div>
                    <div class="form-group row" style="margin-bottom:0px">
                      <label for="exampleInputMobile" class="col-sm-3 col-form-label">Pincode</label>
                      <div class="col-sm-9">
                        <input  value="<%=rs3.getString("pincode")%>" readonly  class="form-control" id="exampleInputMobile">
                      </div>
                    </div>
                    
                        <div class="form-group row" style="margin-bottom:0px">
                          <label class="col-sm-3 col-form-label">Select Timing</label>
                          <div class="col-sm-9">
                            <select class="form-control" name="time">
                              <option selected value="10 AM to 12 PM">10 AM to 12 PM</option>
                              <option  value="12 PM to 4 PM">12 PM to 4 PM</option>
                              <option value="4 PM to 8 PM">4 PM to 8 PM</option>
                             
                            </select>
                          </div>
                        </div>
                        <div class="form-group row" style="margin-bottom:0px">
                          <label class="col-sm-3 col-form-label" >Select Date</label>
                          <div class="col-sm-9">
                          <%Date dl = new Date();
              			SimpleDateFormat s1 = new SimpleDateFormat("yyyy-MM-dd");
              			%>
                            <input type="date" name="date" min="<%=s1.format(dl)%>">
                          </div>
                        </div>
                       
                    
                   
                   <input type="hidden" name="bookid" value="<%=request.getParameter("bookid")%>">
                                  <input type="hidden" name="email" value="<%=request.getParameter("email")%>">
                                  <input type="hidden" name="time" value="<%=request.getParameter("time")%>">
                                  <input type="hidden" name="date" value="<%=request.getParameter("date")%>">
               
                    <button type="submit" name="action" value="contract" class="btn btn-primary mr-2">Confirm Contract</button>
                    <button class="btn btn-light">Cancel</button>
                  </form>
                </div>
                
                
<%} 
										
									
										cn3.close();
										%>
										<%
										}
										cn2.close();%>
						
					

				</div>
<jsp:include page="footer.jsp" />



			
</body>

</html>

